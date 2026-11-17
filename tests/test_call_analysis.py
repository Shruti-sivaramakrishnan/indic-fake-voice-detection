import numpy as np
import pytest

from call_analysis import (
    SpeechRegion,
    analyze_call,
    build_analysis_windows,
    detect_speech_regions,
)
from voice_guard_core import (
    AudioInputPolicy,
    CallInputPolicy,
    validate_encoded_audio_size,
)


def make_tone(duration, sample_rate=16000, amplitude=0.08):
    time = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    return amplitude * np.sin(2 * np.pi * 220 * time)


def make_two_section_recording():
    first = make_tone(6.0)
    silence = np.zeros(int(0.8 * 16000), dtype=np.float32)
    second = make_tone(6.0)
    return np.concatenate([first, silence, second])


def patch_call_pipeline(monkeypatch, waveform, probabilities):
    import model_pipeline

    monkeypatch.setattr(
        model_pipeline, "decode_audio_bytes",
        lambda audio_bytes: (waveform, 16000),
    )
    monkeypatch.setattr(
        model_pipeline, "ensure_target_sample_rate",
        lambda samples, original_rate, target_rate: samples,
    )
    monkeypatch.setattr(
        model_pipeline, "load_inference_bundle",
        lambda **kwargs: object(),
    )

    scores = iter(probabilities)
    monkeypatch.setattr(
        model_pipeline, "score_waveform",
        lambda segment, sample_rate, bundle=None: (0.0, next(scores)),
    )


def test_silence_produces_no_speech_regions():
    silence = np.zeros(5 * 16000, dtype=np.float32)
    assert detect_speech_regions(silence, 16000) == []


def test_quiet_speech_is_detected():
    quiet_speech = make_tone(2.0, amplitude=0.003)
    regions = detect_speech_regions(quiet_speech, 16000)
    assert len(regions) == 1
    assert regions[0].end_seconds - regions[0].start_seconds >= 1.8


def test_silence_gap_creates_separate_speech_regions():
    waveform = np.concatenate([
        make_tone(2.0),
        np.zeros(int(0.8 * 16000), dtype=np.float32),
        make_tone(2.0),
    ])
    regions = detect_speech_regions(waveform, 16000)
    assert len(regions) == 2
    assert regions[0].end_seconds < regions[1].start_seconds
    assert 0.5 <= regions[1].start_seconds - regions[0].end_seconds <= 1.1


def test_windows_use_six_seconds_and_three_second_hop():
    windows = build_analysis_windows([SpeechRegion(0.0, 13.0)], 16000)
    assert [window.start_seconds for window in windows] == [0.0, 3.0, 6.0, 9.0]
    assert windows[0].end_seconds == 6.0
    assert windows[1].end_seconds == 9.0


def test_trailing_window_shorter_than_two_seconds_is_dropped():
    windows = build_analysis_windows([SpeechRegion(0.0, 7.0)], 16000)
    assert all(window.end_seconds - window.start_seconds >= 2.0 for window in windows)
    assert len(windows) == 2
    assert windows[-1].start_seconds == 3.0
    assert windows[-1].end_seconds == 7.0


def test_call_and_single_clip_upload_policies_remain_separate():
    assert CallInputPolicy().max_bytes == 200 * 1024 * 1024
    assert CallInputPolicy().max_duration_seconds == 900.0
    assert AudioInputPolicy().max_bytes == 25 * 1024 * 1024
    assert AudioInputPolicy().max_duration_seconds == 30.0


def test_call_upload_size_limit_is_enforced():
    from dataclasses import replace
    from voice_guard_core import AudioValidationError

    small_policy = replace(CallInputPolicy(), max_bytes=10)
    with pytest.raises(AudioValidationError, match="too large"):
        validate_encoded_audio_size(b"x" * 11, policy=small_policy)


def test_recording_over_fifteen_minutes_is_rejected(monkeypatch):
    import model_pipeline

    monkeypatch.setattr(
        model_pipeline, "decode_audio_bytes",
        lambda audio_bytes: (np.zeros(901, dtype=np.float32), 1),
    )

    def unexpected_resampling(*args, **kwargs):
        raise AssertionError("Resampling should not run for an overlong call.")

    monkeypatch.setattr(
        model_pipeline, "ensure_target_sample_rate",
        unexpected_resampling,
    )

    with pytest.raises(ValueError, match="15 minutes"):
        analyze_call(b"test-audio")


def test_mixed_call_reports_flagged_range_and_progress(monkeypatch):
    waveform = make_two_section_recording()
    patch_call_pipeline(monkeypatch, waveform, [0.1, 0.1, 0.8, 0.8])

    progress = []
    report = analyze_call(b"test-audio", progress_callback=progress.append)

    assert report.windows_analysed == 4
    assert report.flagged_windows == 2
    assert report.flagged_share == pytest.approx(0.5)
    assert len(report.flagged_ranges) == 1
    assert report.flagged_ranges[0].start_seconds > 6.5
    assert report.flagged_ranges[0].end_seconds > report.flagged_ranges[0].start_seconds
    assert report.highest_p_fake == pytest.approx(0.8)
    assert report.highest_score_timestamp == report.window_scores[2].start_seconds
    assert progress == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_all_real_call_has_no_flagged_ranges(monkeypatch):
    waveform = make_two_section_recording()
    patch_call_pipeline(monkeypatch, waveform, [0.1, 0.1, 0.2, 0.2])

    report = analyze_call(b"test-audio")

    assert report.windows_analysed == 4
    assert report.flagged_windows == 0
    assert report.flagged_share == 0.0
    assert report.flagged_ranges == []

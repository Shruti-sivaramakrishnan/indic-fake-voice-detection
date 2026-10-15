import hashlib
from types import SimpleNamespace

import pytest

import codec_robustness as cr


def model_result(audio_bytes: bytes, p_fake: float, verdict: str, sample_rate: int = 16000):
    return SimpleNamespace(
        verdict=verdict,
        audio_hash=hashlib.sha256(audio_bytes).hexdigest(),
        metadata={
            "p_fake_calibrated": p_fake,
            "sample_rate_in": sample_rate,
        },
    )


def test_find_ffmpeg_reports_missing_dependency(monkeypatch) -> None:
    monkeypatch.setattr(cr.shutil, "which", lambda _name: None)
    with pytest.raises(cr.CodecRobustnessError, match="FFmpeg is required"):
        cr.find_ffmpeg()


def test_transcode_uses_fixed_pipe_command(monkeypatch) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"encoded", stderr=b"")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    output = cr.transcode_audio(
        b"source audio",
        cr.DEFAULT_CONDITIONS[0],
        ffmpeg_path="/test/ffmpeg",
    )

    assert output == b"encoded"
    assert captured["input"] == b"source audio"
    assert captured["command"][0] == "/test/ffmpeg"
    assert captured["command"][-1] == "pipe:1"
    assert "libopus" in captured["command"]
    assert captured["timeout"] == 60.0


def test_report_calculates_deltas_and_keeps_partial_failures(monkeypatch) -> None:
    source = b"original valid audio"
    original = model_result(source, 0.40, "uncertain")

    transformed = {
        "opus_16kbps": b"opus",
        "g711_alaw_8khz": b"g711",
    }

    def fake_transcode(_audio, condition, **_kwargs):
        if condition.condition_id == "telephone_pcm_8khz":
            raise cr.CodecRobustnessError("telephone transform failed")
        return transformed[condition.condition_id]

    def fake_analyze(audio_bytes):
        if audio_bytes == b"opus":
            return model_result(audio_bytes, 0.55, "uncertain", 48000)
        if audio_bytes == b"g711":
            return model_result(audio_bytes, 0.72, "fake", 8000)
        raise AssertionError("unexpected analysis input")

    monkeypatch.setattr(cr, "transcode_audio", fake_transcode)
    report = cr.analyze_codec_robustness(
        source,
        fake_analyze,
        original_result=original,
        ffmpeg_path="/test/ffmpeg",
    )

    assert [row.condition_id for row in report.conditions] == [
        "original",
        "opus_16kbps",
        "g711_alaw_8khz",
        "telephone_pcm_8khz",
    ]
    assert report.conditions[1].delta_p_fake == pytest.approx(0.15)
    assert report.conditions[1].verdict_changed is False
    assert report.conditions[2].delta_p_fake == pytest.approx(0.32)
    assert report.conditions[2].verdict_changed is True
    assert report.conditions[3].error == "telephone transform failed"
    assert report.transformed_success_count == 2
    assert report.verdict_change_count == 1
    assert report.max_absolute_delta == pytest.approx(0.32)


def test_existing_result_must_match_selected_audio() -> None:
    source = b"selected audio"
    stale = model_result(b"different audio", 0.2, "real")

    with pytest.raises(cr.CodecRobustnessError, match="different audio file"):
        cr.analyze_codec_robustness(
            source,
            lambda _audio: None,
            original_result=stale,
            ffmpeg_path="/test/ffmpeg",
        )

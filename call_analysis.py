"""Long-call speech detection and window segmentation for Voice Guard."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Callable, List

import numpy as np


@dataclass(frozen=True)
class SpeechRegion:
    """Speech-active interval in the original recording timeline."""

    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class AnalysisWindow:
    """A model-analysis window with its original recording timestamps."""

    start_seconds: float
    end_seconds: float
    start_sample: int
    end_sample: int


def detect_speech_regions(
    waveform: np.ndarray,
    sample_rate: int,
    frame_duration: float = 0.025,
    hop_duration: float = 0.010,
    merge_gap_seconds: float = 0.30,
    min_region_seconds: float = 0.15,
    energy_threshold: float | None = None,
) -> List[SpeechRegion]:
    """Find speech-active regions using a lightweight adaptive energy gate.

    This is an energy-based detector, not a trained voice activity detector.
    It preserves timestamps relative to the original recording.
    """
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if samples.size == 0:
        return []
    if not np.isfinite(samples).all():
        raise ValueError("waveform contains non-finite values.")
    if frame_duration <= 0 or hop_duration <= 0:
        raise ValueError("Frame and hop durations must be positive.")

    frame_size = max(1, int(round(sample_rate * frame_duration)))
    hop_size = max(1, int(round(sample_rate * hop_duration)))

    if samples.size < frame_size:
        starts = [0]
    else:
        starts = range(0, samples.size - frame_size + 1, hop_size)

    frame_rms = np.asarray(
        [
            np.sqrt(np.mean(
                np.square(samples[start : start + frame_size], dtype=np.float64)
            ))
            for start in starts
        ],
        dtype=np.float64,
    )

    if energy_threshold is None:
        noise_floor = float(np.percentile(frame_rms, 20))
        peak_frame = float(np.max(frame_rms))
        threshold = max(1e-4, min(noise_floor * 2.0, peak_frame * 0.10))
    else:
        if energy_threshold < 0:
            raise ValueError("energy_threshold cannot be negative.")
        threshold = float(energy_threshold)

    active = frame_rms >= threshold
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        return []

    merge_frames = int(round(merge_gap_seconds / hop_duration))
    groups = []
    group_start = previous = int(active_indices[0])

    for index in active_indices[1:]:
        index = int(index)
        if index - previous - 1 > merge_frames:
            groups.append((group_start, previous))
            group_start = index
        previous = index
    groups.append((group_start, previous))

    regions = []
    for first_frame, last_frame in groups:
        start_sample = first_frame * hop_size
        end_sample = min(samples.size, last_frame * hop_size + frame_size)
        start_seconds = start_sample / float(sample_rate)
        end_seconds = end_sample / float(sample_rate)

        if end_seconds - start_seconds >= min_region_seconds:
            regions.append(
                SpeechRegion(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
            )

    return regions


def build_analysis_windows(
    regions: List[SpeechRegion],
    sample_rate: int,
    window_seconds: float = 6.0,
    hop_seconds: float = 3.0,
    min_trailing_seconds: float = 2.0,
) -> List[AnalysisWindow]:
    """Create overlapping analysis windows within speech regions."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if window_seconds <= 0 or hop_seconds <= 0 or min_trailing_seconds < 0:
        raise ValueError("Window, hop, and trailing durations must be valid.")

    windows = []
    for region in regions:
        if region.start_seconds < 0 or region.end_seconds < region.start_seconds:
            raise ValueError("Invalid speech region.")

        start = region.start_seconds
        while start < region.end_seconds:
            duration = min(window_seconds, region.end_seconds - start)
            if duration < min_trailing_seconds:
                break

            start_sample = int(round(start * sample_rate))
            end_sample = int(round((start + duration) * sample_rate))

            windows.append(
                AnalysisWindow(
                    start_seconds=start,
                    end_seconds=start + duration,
                    start_sample=start_sample,
                    end_sample=end_sample,
                )
            )
            start += hop_seconds

    return windows


@dataclass(frozen=True)
class WindowScore:
    """Fake-voice score for one analysis window."""

    start_seconds: float
    end_seconds: float
    raw_logit: float
    p_fake: float
    verdict: str


@dataclass(frozen=True)
class CallAnalysisReport:
    """Complete analysis results for one phone-call recording."""

    duration_seconds: float
    speech_duration_seconds: float
    windows_analysed: int
    flagged_windows: int
    flagged_share: float
    highest_p_fake: float
    highest_score_timestamp: float
    window_scores: List[WindowScore]
    flagged_ranges: List[SpeechRegion]


def merge_flagged_ranges(
    window_scores: List[WindowScore],
) -> List[SpeechRegion]:
    """Merge overlapping or adjacent windows classified as fake."""
    flagged = sorted(
        (score for score in window_scores if score.verdict == "fake"),
        key=lambda score: score.start_seconds,
    )

    if not flagged:
        return []

    merged = []
    current_start = flagged[0].start_seconds
    current_end = flagged[0].end_seconds

    for score in flagged[1:]:
        if score.start_seconds <= current_end:
            current_end = max(current_end, score.end_seconds)
        else:
            merged.append(
                SpeechRegion(current_start, current_end)
            )
            current_start = score.start_seconds
            current_end = score.end_seconds

    merged.append(SpeechRegion(current_start, current_end))
    return merged


def analyze_call(
    audio_bytes: bytes,
    checkpoint_dir: str | None = None,
    window_seconds: float = 6.0,
    hop_seconds: float = 3.0,
    progress_callback: Callable[[float], None] | None = None,
    input_policy=None,
) -> CallAnalysisReport:
    """Analyse a long phone call using the existing Voice Guard model."""
    import model_pipeline

    from voice_guard_core import (
        AudioValidationError,
        CallInputPolicy,
        validate_encoded_audio_size,
    )

    policy = input_policy or CallInputPolicy()
    validate_encoded_audio_size(
        audio_bytes,
        policy=policy,
    )

    waveform, original_sample_rate = model_pipeline.decode_audio_bytes(
        audio_bytes
    )

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)

    if waveform.size == 0:
        raise ValueError("The decoded recording is empty.")
    if original_sample_rate <= 0:
        raise ValueError("Invalid audio sample rate.")
    if not np.isfinite(waveform).all():
        raise ValueError("The recording contains non-finite audio values.")

    duration_seconds = waveform.size / float(original_sample_rate)

    if duration_seconds > policy.max_duration_seconds:
        raise AudioValidationError(
            "The call is too long. Upload a recording of 15 minutes or less."
        )
    if duration_seconds < policy.min_duration_seconds:
        raise AudioValidationError(
            "The call is too short. Call analysis requires at least 30 seconds. "
            "Use Single Clip mode for shorter recordings."
        )

    waveform = model_pipeline.ensure_target_sample_rate(
        waveform,
        original_sample_rate,
        model_pipeline.TARGET_SAMPLE_RATE,
    )
    sample_rate = model_pipeline.TARGET_SAMPLE_RATE

    regions = detect_speech_regions(waveform, sample_rate)
    windows = build_analysis_windows(
        regions,
        sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )

    if not windows:
        raise AudioValidationError(
            "No speech-active regions long enough for analysis were detected."
        )

    if checkpoint_dir is None:
        bundle = model_pipeline.load_inference_bundle()
    else:
        bundle = model_pipeline.load_inference_bundle(
            checkpoint_dir=checkpoint_dir
        )

    window_scores = []

    for index, window in enumerate(windows):
        scoring_start = window.start_seconds
        scoring_end = window.end_seconds
        segment_duration = scoring_end - scoring_start

        # Expand short Call Analysis windows using surrounding audio.
        # Single Clip and the shared model pipeline are not modified.
        if segment_duration < window_seconds and duration_seconds >= window_seconds:
            center = (scoring_start + scoring_end) / 2.0
            scoring_start = max(
                0.0,
                min(
                    center - window_seconds / 2.0,
                    duration_seconds - window_seconds,
                ),
            )
            scoring_end = scoring_start + window_seconds

        start_sample = int(round(scoring_start * sample_rate))
        end_sample = int(round(scoring_end * sample_rate))
        segment = waveform[start_sample:end_sample]

        raw_logit, p_fake = model_pipeline.score_waveform(
            segment,
            sample_rate,
            bundle=bundle,
        )
        verdict = model_pipeline.verdict_from_p_fake(p_fake)

        window_scores.append(
            WindowScore(
                # Keep the original speech-region timestamps even if the audio
                # used for scoring was expanded to a full six seconds.
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
                raw_logit=raw_logit,
                p_fake=p_fake,
                verdict=verdict,
            )
        )

        if progress_callback is not None:
            progress_callback((index + 1) / len(windows))

    flagged_windows = sum(
        score.verdict == "fake" for score in window_scores
    )

    flagged_share = flagged_windows / len(window_scores)

    highest_score = max(window_scores, key=lambda score: score.p_fake)

    speech_duration_seconds = sum(
        region.end_seconds - region.start_seconds
        for region in regions
    )

    return CallAnalysisReport(
        duration_seconds=duration_seconds,
        speech_duration_seconds=speech_duration_seconds,
        windows_analysed=len(window_scores),
        flagged_windows=flagged_windows,
        flagged_share=flagged_share,
        highest_p_fake=highest_score.p_fake,
        highest_score_timestamp=highest_score.start_seconds,
        window_scores=window_scores,
        flagged_ranges=merge_flagged_ranges(window_scores),
    )


def report_to_csv_bytes(report: CallAnalysisReport) -> bytes:
    """Serialize one row per analysed window without writing audio or temp files."""
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        ["window", "start_seconds", "end_seconds", "p_fake", "verdict"]
    )
    for index, score in enumerate(report.window_scores, start=1):
        writer.writerow(
            [
                index,
                f"{score.start_seconds:.3f}",
                f"{score.end_seconds:.3f}",
                f"{score.p_fake:.8f}",
                score.verdict,
            ]
        )
    return output.getvalue().encode("utf-8")

"""Offline, dependency-light speaker clustering for call-review assistance.

The implementation intentionally assigns anonymous labels (``Speaker A`` and
``Speaker B``).  It is not speaker identification and must not be used to infer
who a person is.  A stronger diarization backend can be injected without
changing transcript alignment or per-speaker conversation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from conversation_analysis import ConversationReport, TranscriptSegment, analyze_transcript


TARGET_SAMPLE_RATE = 16_000


class DiarizationError(RuntimeError):
    """A safe, user-displayable diarization failure."""


@dataclass(frozen=True)
class SpeakerTurn:
    start_seconds: float
    end_seconds: float
    speaker_label: str
    confidence: float


@dataclass(frozen=True)
class SpeakerTranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str
    speaker_label: str
    confidence: float


@dataclass(frozen=True)
class SpeakerSummary:
    speaker_label: str
    segment_count: int
    speaking_seconds: float
    conversation: ConversationReport


@dataclass(frozen=True)
class SpeakerAwareReport:
    backend_id: str
    turns: tuple[SpeakerTurn, ...]
    segments: tuple[SpeakerTranscriptSegment, ...]
    speakers: tuple[SpeakerSummary, ...]


def _frame_features(frame: np.ndarray, sample_rate: int) -> np.ndarray:
    windowed = frame * np.hanning(frame.size)
    magnitude = np.abs(np.fft.rfft(windowed)) + 1e-8
    frequencies = np.fft.rfftfreq(frame.size, 1.0 / sample_rate)
    total = float(magnitude.sum())
    centroid = float((frequencies * magnitude).sum() / total) / (sample_rate / 2)
    spread = float(
        np.sqrt((((frequencies - centroid * sample_rate / 2) ** 2) * magnitude).sum() / total)
    ) / (sample_rate / 2)
    band_edges = np.linspace(0, magnitude.size, 7, dtype=int)
    band_energy = [
        np.log(float(magnitude[band_edges[i] : band_edges[i + 1]].sum()) + 1e-8)
        for i in range(6)
    ]
    zero_crossing = float(np.mean(np.abs(np.diff(np.signbit(frame)))))
    return np.asarray([centroid, spread, zero_crossing, *band_energy], dtype=np.float64)


def _two_cluster(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic two-means with a one-speaker fallback."""
    if features.shape[0] < 3:
        return np.zeros(features.shape[0], dtype=int), np.ones(features.shape[0])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    normalized = (features - mean) / np.where(scale < 1e-6, 1.0, scale)
    distances = ((normalized[:, None, :] - normalized[None, :, :]) ** 2).sum(axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    centers = np.stack([normalized[first], normalized[second]])
    labels = np.zeros(normalized.shape[0], dtype=int)
    for _ in range(25):
        center_distances = ((normalized[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = center_distances.argmin(axis=1)
        if np.array_equal(labels, new_labels) and _ > 0:
            break
        labels = new_labels
        for cluster in (0, 1):
            members = normalized[labels == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)

    separation = float(np.linalg.norm(centers[0] - centers[1]))
    if separation < 1.15 or min(np.bincount(labels, minlength=2)) == 0:
        return np.zeros(features.shape[0], dtype=int), np.full(features.shape[0], 0.55)

    # Stable labels: the lower spectral centroid is always Speaker A.
    raw_centroids = [features[labels == cluster, 0].mean() for cluster in (0, 1)]
    if raw_centroids[0] > raw_centroids[1]:
        labels = 1 - labels
        centers = centers[::-1]
    center_distances = np.sqrt(
        ((normalized[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    )
    ordered = np.sort(center_distances, axis=1)
    margin = (ordered[:, 1] - ordered[:, 0]) / (ordered[:, 1] + 1e-8)
    confidence = np.clip(0.5 + 0.5 * margin, 0.5, 0.99)
    return labels, confidence


def diarize_waveform(
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    frame_seconds: float = 1.5,
    hop_seconds: float = 0.75,
) -> tuple[SpeakerTurn, ...]:
    """Cluster voiced frames and merge adjacent frames with the same label."""
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if samples.size == 0 or sample_rate <= 0 or not np.isfinite(samples).all():
        raise DiarizationError("The recording could not be prepared for speaker analysis.")
    frame_size = max(400, int(frame_seconds * sample_rate))
    hop_size = max(160, int(hop_seconds * sample_rate))
    if samples.size < frame_size:
        padded = np.zeros(frame_size, dtype=np.float32)
        padded[: samples.size] = samples
        samples = padded

    starts = list(range(0, max(1, samples.size - frame_size + 1), hop_size))
    last_start = max(0, samples.size - frame_size)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    frames = np.stack([samples[start : start + frame_size] for start in starts])
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(rms.max(initial=0.0))
    if peak < 1e-5:
        raise DiarizationError("No audible speech was found for speaker analysis.")
    threshold = max(1e-5, min(float(np.percentile(rms, 60)) * 0.5, peak * 0.2))
    voiced_indices = np.flatnonzero(rms >= threshold)
    if voiced_indices.size == 0:
        raise DiarizationError("No audible speech was found for speaker analysis.")

    features = np.stack([_frame_features(frames[index], sample_rate) for index in voiced_indices])
    labels, confidence = _two_cluster(features)
    raw_turns = []
    duration = samples.size / float(sample_rate)
    for frame_index, label, score in zip(voiced_indices, labels, confidence):
        start = starts[int(frame_index)] / float(sample_rate)
        end = min(duration, (starts[int(frame_index)] + frame_size) / float(sample_rate))
        raw_turns.append(SpeakerTurn(start, end, f"Speaker {chr(65 + int(label))}", float(score)))

    merged: list[SpeakerTurn] = []
    for turn in raw_turns:
        if (
            merged
            and merged[-1].speaker_label == turn.speaker_label
            and turn.start_seconds <= merged[-1].end_seconds + hop_seconds
        ):
            previous = merged[-1]
            merged[-1] = SpeakerTurn(
                previous.start_seconds,
                max(previous.end_seconds, turn.end_seconds),
                previous.speaker_label,
                (previous.confidence + turn.confidence) / 2,
            )
        else:
            merged.append(turn)
    return tuple(merged)


def diarize_audio_bytes(
    audio_bytes: bytes,
    backend: Callable[[np.ndarray, int], Iterable[SpeakerTurn]] | None = None,
) -> tuple[SpeakerTurn, ...]:
    """Decode audio and run either the built-in or an injected backend."""
    import model_pipeline
    from voice_guard_core import CallInputPolicy, validate_encoded_audio_size

    policy = CallInputPolicy()
    try:
        validate_encoded_audio_size(audio_bytes, policy=policy)
        waveform, sample_rate = model_pipeline.decode_audio_bytes(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError("The recording could not be decoded for speaker analysis.") from exc
    try:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        duration = waveform.size / float(sample_rate) if sample_rate > 0 else 0.0
        if duration < policy.min_duration_seconds or duration > policy.max_duration_seconds:
            raise DiarizationError(
                "Speaker analysis supports calls from 30 seconds to 15 minutes."
            )
        waveform = model_pipeline.ensure_target_sample_rate(
            waveform, sample_rate, TARGET_SAMPLE_RATE
        )
        runner = backend or diarize_waveform
        turns = tuple(runner(waveform, TARGET_SAMPLE_RATE))
    except DiarizationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError("Speaker analysis could not be completed for this recording.") from exc
    if not turns:
        raise DiarizationError("No speaker turns were produced for this recording.")
    return turns


def align_transcript_to_speakers(
    segments: Iterable[TranscriptSegment], turns: Iterable[SpeakerTurn]
) -> tuple[SpeakerTranscriptSegment, ...]:
    turn_list = tuple(turns)
    aligned = []
    for segment in segments:
        overlaps = [
            max(0.0, min(segment.end_seconds, turn.end_seconds) - max(segment.start_seconds, turn.start_seconds))
            for turn in turn_list
        ]
        if overlaps and max(overlaps) > 0:
            selected = turn_list[int(np.argmax(overlaps))]
            label, score = selected.speaker_label, selected.confidence
        else:
            label, score = "Unassigned", 0.0
        aligned.append(
            SpeakerTranscriptSegment(
                segment.start_seconds, segment.end_seconds, segment.text, label, score
            )
        )
    return tuple(aligned)


def build_speaker_aware_report(
    segments: Iterable[TranscriptSegment],
    turns: Iterable[SpeakerTurn],
    backend_id: str = "voiceguard-lightweight-v1",
) -> SpeakerAwareReport:
    turn_list = tuple(turns)
    aligned = align_transcript_to_speakers(segments, turn_list)
    summaries = []
    for label in sorted({segment.speaker_label for segment in aligned}):
        selected = [segment for segment in aligned if segment.speaker_label == label]
        report_segments = [
            TranscriptSegment(segment.start_seconds, segment.end_seconds, segment.text)
            for segment in selected
        ]
        summaries.append(
            SpeakerSummary(
                label,
                len(selected),
                sum(segment.end_seconds - segment.start_seconds for segment in selected),
                analyze_transcript(report_segments),
            )
        )
    return SpeakerAwareReport(backend_id, turn_list, aligned, tuple(summaries))

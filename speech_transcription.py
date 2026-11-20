"""Multilingual speech-to-text orchestration for call conversation analysis.

The ASR model is loaded lazily because it is independent from the synthetic-
voice detector. Audio and transcripts remain in memory; this module does not
write either artifact to disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np

from conversation_analysis import (
    ConversationReport,
    TranscriptSegment,
    analyze_transcript,
)


DEFAULT_ASR_MODEL = "openai/whisper-base"
TARGET_SAMPLE_RATE = 16_000


class TranscriptionError(RuntimeError):
    """A safe, user-displayable transcription failure."""


@dataclass(frozen=True)
class TranscriptionResult:
    model_id: str
    duration_seconds: float
    segments: tuple[TranscriptSegment, ...]

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments).strip()


@dataclass(frozen=True)
class AudioConversationResult:
    transcription: TranscriptionResult
    conversation: ConversationReport


def _asr_device() -> tuple[object, object]:
    import torch

    requested = os.getenv("VOICE_GUARD_ASR_DEVICE", "").strip()
    if requested:
        device = requested
    elif torch.cuda.is_available():
        device = 0
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1
    dtype = torch.float16 if device == 0 else torch.float32
    return device, dtype


@lru_cache(maxsize=2)
def load_asr_pipeline(model_id: str | None = None):
    """Load and cache a timestamp-capable multilingual Whisper pipeline."""
    from transformers import pipeline

    selected_model = model_id or os.getenv(
        "VOICE_GUARD_ASR_MODEL", DEFAULT_ASR_MODEL
    )
    device, dtype = _asr_device()
    return pipeline(
        "automatic-speech-recognition",
        model=selected_model,
        device=device,
        dtype=dtype,
        chunk_length_s=30,
    )


def _timestamp_pair(value, duration_seconds: float) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    raw_start, raw_end = value
    if raw_start is None:
        return None
    start = max(0.0, min(float(raw_start), duration_seconds))
    end = duration_seconds if raw_end is None else float(raw_end)
    end = max(start, min(end, duration_seconds))
    if end <= start:
        end = min(duration_seconds, start + 0.01)
    return (start, end) if end > start else None


def segments_from_pipeline_output(
    output: dict, duration_seconds: float
) -> tuple[TranscriptSegment, ...]:
    """Convert Transformers ASR output to validated timestamped segments."""
    segments = []
    for chunk in output.get("chunks") or []:
        text = str(chunk.get("text") or "").strip()
        interval = _timestamp_pair(chunk.get("timestamp"), duration_seconds)
        if text and interval is not None:
            segments.append(TranscriptSegment(interval[0], interval[1], text))

    if not segments:
        text = str(output.get("text") or "").strip()
        if text and duration_seconds > 0:
            segments.append(TranscriptSegment(0.0, duration_seconds, text))

    segments.sort(key=lambda segment: (segment.start_seconds, segment.end_seconds))
    return tuple(segments)


def transcribe_audio_bytes(
    audio_bytes: bytes,
    model_id: str | None = None,
    asr_pipeline=None,
) -> TranscriptionResult:
    """Decode, validate, resample and transcribe one supported call recording."""
    import model_pipeline
    from voice_guard_core import CallInputPolicy, validate_encoded_audio_size

    policy = CallInputPolicy()
    validate_encoded_audio_size(audio_bytes, policy=policy)
    waveform, sample_rate = model_pipeline.decode_audio_bytes(audio_bytes)
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0 or sample_rate <= 0 or not np.isfinite(waveform).all():
        raise TranscriptionError("The recording could not be prepared for transcription.")

    duration_seconds = waveform.size / float(sample_rate)
    if duration_seconds < policy.min_duration_seconds:
        raise TranscriptionError("Call transcription requires at least 30 seconds of audio.")
    if duration_seconds > policy.max_duration_seconds:
        raise TranscriptionError("Call transcription supports recordings up to 15 minutes.")

    waveform = model_pipeline.ensure_target_sample_rate(
        waveform, sample_rate, TARGET_SAMPLE_RATE
    )
    selected_model = model_id or os.getenv(
        "VOICE_GUARD_ASR_MODEL", DEFAULT_ASR_MODEL
    )
    recognizer = asr_pipeline or load_asr_pipeline(selected_model)
    try:
        output = recognizer(
            {"array": waveform, "sampling_rate": TARGET_SAMPLE_RATE},
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"},
        )
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(
            "Automatic transcription failed. Use the timestamped transcript CSV fallback."
        ) from exc

    segments = segments_from_pipeline_output(output, duration_seconds)
    if not segments:
        raise TranscriptionError(
            "No transcript was produced. Check that the call contains clear speech."
        )
    return TranscriptionResult(selected_model, duration_seconds, segments)


def analyze_audio_conversation(
    audio_bytes: bytes,
    transcriber: Callable[[bytes], TranscriptionResult] | None = None,
) -> AudioConversationResult:
    """Run the complete audio to transcript to conversation-cue workflow."""
    transcription = (
        transcriber(audio_bytes)
        if transcriber is not None
        else transcribe_audio_bytes(audio_bytes)
    )
    return AudioConversationResult(
        transcription=transcription,
        conversation=analyze_transcript(transcription.segments),
    )

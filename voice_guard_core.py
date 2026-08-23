"""Dependency-light safety and decision helpers for Voice Guard.

This module deliberately does not import torch or Streamlit.  Keeping the input
policy and result policy here makes them cheap to unit-test and lets the web UI
validate an upload before the multi-gigabyte frontend is loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


SUPPORTED_LANGUAGES = (
    "Assamese",
    "Bengali",
    "Bodo",
    "Dogri",
    "English",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Malayalam",
    "Manipuri",
    "Marathi",
    "Nepali",
    "Odia",
    "Sanskrit",
    "Tamil",
    "Telugu",
)


class AudioValidationError(ValueError):
    """A safe, user-displayable rejection of unsuitable input audio."""


@dataclass(frozen=True)
class AudioInputPolicy:
    """Limits that protect both result quality and quadratic GAT inference."""

    max_bytes: int = 25 * 1024 * 1024
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 30.0
    min_rms: float = 1e-4
    min_peak: float = 1e-3
    min_active_seconds: float = 0.45


@dataclass(frozen=True)
class DecisionPolicy:
    """Three-way abstention policy operating on calibrated P(fake)."""

    policy_id: str = "legacy-symmetric-abstention-v1"
    real_max_p_fake: float = 0.35
    fake_min_p_fake: float = 0.65
    provenance: str = (
        "Legacy symmetric abstention band; not selected on the frozen test set. "
        "Replace only from validation data under the v2 protocol."
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.real_max_p_fake < self.fake_min_p_fake <= 1.0:
            raise ValueError("Decision thresholds must satisfy 0 <= real < fake <= 1.")


DEFAULT_AUDIO_POLICY = AudioInputPolicy()
DEFAULT_DECISION_POLICY = DecisionPolicy()


def validate_encoded_audio_size(
    audio_bytes: bytes,
    policy: AudioInputPolicy = DEFAULT_AUDIO_POLICY,
) -> None:
    if not isinstance(audio_bytes, (bytes, bytearray)):
        raise AudioValidationError("The uploaded audio could not be read.")
    if not audio_bytes:
        raise AudioValidationError("The uploaded audio file is empty.")
    if len(audio_bytes) > policy.max_bytes:
        limit_mb = policy.max_bytes / (1024 * 1024)
        raise AudioValidationError(
            f"The audio file is too large. Upload a file no larger than {limit_mb:.0f} MB."
        )


def validate_decoded_audio(
    waveform: np.ndarray,
    sample_rate: int,
    policy: AudioInputPolicy = DEFAULT_AUDIO_POLICY,
) -> Dict[str, Any]:
    """Validate decoded mono audio and return transparent quality measurements."""
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)

    if sample_rate <= 0:
        raise AudioValidationError("The audio file has an invalid sample rate.")
    if samples.size == 0:
        raise AudioValidationError("The uploaded audio contains no samples.")
    if not np.isfinite(samples).all():
        raise AudioValidationError("The audio contains invalid sample values.")

    duration = samples.size / float(sample_rate)
    if duration < policy.min_duration_seconds:
        raise AudioValidationError(
            f"The clip is too short. Use at least {policy.min_duration_seconds:.0f} second "
            "of clear speech."
        )
    if duration > policy.max_duration_seconds:
        raise AudioValidationError(
            f"The clip is too long. Use a clip of {policy.max_duration_seconds:.0f} seconds "
            "or less."
        )

    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if peak < policy.min_peak or rms < policy.min_rms:
        raise AudioValidationError(
            "No clear speech was detected. Upload a recording with an audible voice."
        )

    # A lightweight energy gate catches digital silence and clips with only a
    # short click. It is intentionally conservative: this is not a language- or
    # speaker-dependent VAD and should not reject quiet but usable speech.
    frame_size = max(1, int(round(sample_rate * 0.025)))
    hop_size = max(1, int(round(sample_rate * 0.010)))
    if samples.size <= frame_size:
        frame_starts = [0]
    else:
        frame_starts = range(0, samples.size - frame_size + 1, hop_size)
    frame_rms = np.asarray(
        [
            np.sqrt(np.mean(np.square(samples[start : start + frame_size], dtype=np.float64)))
            for start in frame_starts
        ],
        dtype=np.float64,
    )
    active_threshold = max(policy.min_rms * 4.0, rms * 0.35)
    active_frames = int(np.count_nonzero(frame_rms >= active_threshold))
    active_seconds = active_frames * hop_size / float(sample_rate)
    if active_seconds < policy.min_active_seconds:
        raise AudioValidationError(
            "Not enough clear speech was detected. Use a clip with at least half a second "
            "of audible speech."
        )

    clipped_fraction = float(np.mean(np.abs(samples) >= 0.999))
    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": int(sample_rate),
        "rms": rms,
        "peak": peak,
        "active_speech_seconds_estimate": round(active_seconds, 3),
        "clipped_fraction": clipped_fraction,
        "quality_warning": (
            "possible_clipping" if clipped_fraction > 0.01 else None
        ),
    }


def stable_sigmoid(value: float) -> float:
    """Numerically stable scalar sigmoid."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def verdict_from_probability(
    p_fake: float,
    policy: DecisionPolicy = DEFAULT_DECISION_POLICY,
) -> str:
    if not math.isfinite(p_fake) or not 0.0 <= p_fake <= 1.0:
        raise ValueError("p_fake must be a finite probability in [0, 1].")
    if p_fake < policy.real_max_p_fake:
        return "real"
    if p_fake > policy.fake_min_p_fake:
        return "fake"
    return "uncertain"


def normalize_language_hint(language_hint: str | None) -> str:
    """Accept only a declared dataset language; never pretend it was detected."""
    if language_hint is None or not str(language_hint).strip():
        return "unknown"
    value = str(language_hint).strip()
    by_lower = {language.lower(): language for language in SUPPORTED_LANGUAGES}
    try:
        return by_lower[value.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported language hint {value!r}. Choose one of the 16 supported languages."
        ) from exc

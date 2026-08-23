import math

import numpy as np
import pytest

from voice_guard_core import (
    AudioInputPolicy,
    AudioValidationError,
    normalize_language_hint,
    stable_sigmoid,
    validate_decoded_audio,
    validate_encoded_audio_size,
    verdict_from_probability,
)


def test_encoded_audio_rejects_empty_and_oversized() -> None:
    with pytest.raises(AudioValidationError, match="empty"):
        validate_encoded_audio_size(b"")
    with pytest.raises(AudioValidationError, match="too large"):
        validate_encoded_audio_size(b"12345", AudioInputPolicy(max_bytes=4))


def test_decoded_audio_rejects_silence_before_inference() -> None:
    silence = np.zeros(16000, dtype=np.float32)
    with pytest.raises(AudioValidationError, match="No clear speech"):
        validate_decoded_audio(silence, 16000)


def test_decoded_audio_rejects_click_with_no_sustained_speech() -> None:
    waveform = np.zeros(32000, dtype=np.float32)
    waveform[1000:1010] = 0.8
    with pytest.raises(AudioValidationError, match="Not enough clear speech"):
        validate_decoded_audio(waveform, 16000)


def test_decoded_audio_accepts_clear_speech_like_signal() -> None:
    sample_rate = 16000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    waveform = 0.08 * np.sin(2 * np.pi * 220 * time)
    metrics = validate_decoded_audio(waveform, sample_rate)
    assert metrics["duration_seconds"] == 2.0
    assert metrics["active_speech_seconds_estimate"] >= 1.5
    assert metrics["quality_warning"] is None


@pytest.mark.parametrize(
    ("p_fake", "expected"),
    [(0.349, "real"), (0.35, "uncertain"), (0.65, "uncertain"), (0.651, "fake")],
)
def test_verdict_boundaries_are_explicit(p_fake: float, expected: str) -> None:
    assert verdict_from_probability(p_fake) == expected


def test_stable_sigmoid_handles_extreme_logits() -> None:
    assert stable_sigmoid(1000.0) == 1.0
    assert stable_sigmoid(-1000.0) == 0.0
    assert math.isclose(stable_sigmoid(0.0), 0.5)


def test_language_hint_is_user_metadata_only() -> None:
    assert normalize_language_hint(None) == "unknown"
    assert normalize_language_hint("hindi") == "Hindi"
    with pytest.raises(ValueError, match="Unsupported language"):
        normalize_language_hint("Klingon")

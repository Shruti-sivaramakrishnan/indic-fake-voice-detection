import io
import wave

import numpy as np
import pytest

import model_pipeline as mp


def wav_bytes(sample_rate: int = 16000) -> bytes:
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    pcm = (0.1 * np.sin(2 * np.pi * 220 * time) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm.tobytes())
    return output.getvalue()


def test_decode_dataset_audio_from_encoded_bytes() -> None:
    waveform, sample_rate = mp.decode_audio(
        {"audio": {"bytes": wav_bytes(), "path": "unused.wav"}}
    )

    assert sample_rate == 16000
    assert waveform.shape == (16000,)
    assert waveform.dtype == np.float32


def test_decode_dataset_audio_rejects_empty_reference() -> None:
    with pytest.raises(ValueError, match="neither encoded bytes nor a path"):
        mp.decode_audio({"audio": {"bytes": None, "path": None}})

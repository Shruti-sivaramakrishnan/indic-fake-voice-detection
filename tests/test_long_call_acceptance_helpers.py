import numpy as np

from evaluation.run_long_call_acceptance import fill_duration, rotate, wav_bytes


def test_fill_duration_cycles_and_truncates_to_exact_size():
    clips = [np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float32)]
    result = fill_duration(clips, 25 / 16000)
    assert result.shape == (25,)
    assert np.all(result[:10] == 1)
    assert np.all(result[10:20] == 0)
    assert np.all(result[20:] == 1)


def test_rotate_is_deterministic():
    arrays = [np.asarray([1]), np.asarray([2]), np.asarray([3])]
    assert [item.item() for item in rotate(arrays, 1)] == [2, 3, 1]


def test_wav_bytes_round_trip_header():
    payload = wav_bytes(np.zeros(16000, dtype=np.float32))
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"

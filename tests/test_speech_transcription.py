import numpy as np
import pytest

from conversation_analysis import TranscriptSegment
from speech_transcription import (
    TranscriptionError,
    TranscriptionResult,
    analyze_audio_conversation,
    segments_from_pipeline_output,
    transcribe_audio_bytes,
)


def test_pipeline_chunks_become_sorted_clamped_segments():
    output = {
        "text": "ignored aggregate",
        "chunks": [
            {"timestamp": (9.0, None), "text": " send otp now "},
            {"timestamp": (-2.0, 3.0), "text": "hello"},
            {"timestamp": (4.0, 5.0), "text": ""},
        ],
    }
    segments = segments_from_pipeline_output(output, 10.0)
    assert [(s.start_seconds, s.end_seconds, s.text) for s in segments] == [
        (0.0, 3.0, "hello"),
        (9.0, 10.0, "send otp now"),
    ]


def test_aggregate_text_is_a_fallback_segment():
    segments = segments_from_pipeline_output({"text": "clear speech"}, 12.0)
    assert segments == (TranscriptSegment(0.0, 12.0, "clear speech"),)


def test_audio_to_conversation_orchestration_is_end_to_end():
    def fake_transcriber(payload):
        assert payload == b"call"
        return TranscriptionResult(
            "test-asr",
            60.0,
            (
                TranscriptSegment(0, 5, "Thank you"),
                TranscriptSegment(20, 25, "This is urgent send the OTP now"),
            ),
        )

    result = analyze_audio_conversation(b"call", transcriber=fake_transcriber)
    assert result.transcription.model_id == "test-asr"
    assert result.conversation.tone == "positive"
    assert result.conversation.risk_level == "Medium"
    assert result.conversation.category_counts == {
        "credential_or_otp": 1,
        "urgency": 1,
    }


def test_audio_orchestration_includes_speaker_aware_result(monkeypatch):
    import speech_transcription
    from speaker_diarization import SpeakerTurn

    transcription = TranscriptionResult(
        "test-asr",
        60.0,
        (
            TranscriptSegment(0, 5, "Thank you"),
            TranscriptSegment(10, 15, "Urgent, send money and share the OTP"),
        ),
    )
    monkeypatch.setattr(
        speech_transcription,
        "diarize_audio_bytes",
        lambda payload, backend=None: (
            SpeakerTurn(0, 6, "Speaker A", 0.9),
            SpeakerTurn(9, 16, "Speaker B", 0.8),
        ),
    )
    result = analyze_audio_conversation(
        b"call", transcriber=lambda payload: transcription
    )
    assert result.speaker_analysis_error is None
    assert [
        segment.speaker_label for segment in result.speaker_analysis.segments
    ] == ["Speaker A", "Speaker B"]
    assert result.speaker_analysis.speakers[1].conversation.risk_level == "High"


def test_transcribe_validates_and_invokes_injected_pipeline(monkeypatch):
    import model_pipeline

    waveform = np.ones(30 * 8_000, dtype=np.float32) * 0.01
    monkeypatch.setattr(
        model_pipeline, "decode_audio_bytes", lambda payload: (waveform, 8_000)
    )
    monkeypatch.setattr(
        model_pipeline,
        "ensure_target_sample_rate",
        lambda samples, source, target: np.ones(30 * target, dtype=np.float32),
    )
    calls = []

    def fake_asr(value, **kwargs):
        calls.append((value, kwargs))
        return {"chunks": [{"timestamp": (0, 4), "text": "hello"}]}

    result = transcribe_audio_bytes(b"encoded", model_id="fake", asr_pipeline=fake_asr)
    assert result.model_id == "fake"
    assert result.segments[0].text == "hello"
    assert calls[0][0]["sampling_rate"] == 16_000
    assert calls[0][1]["return_timestamps"] is True


def test_empty_asr_result_is_rejected(monkeypatch):
    import model_pipeline

    monkeypatch.setattr(
        model_pipeline,
        "decode_audio_bytes",
        lambda payload: (np.ones(30 * 16_000, dtype=np.float32), 16_000),
    )
    with pytest.raises(TranscriptionError, match="No transcript"):
        transcribe_audio_bytes(
            b"encoded", model_id="fake", asr_pipeline=lambda *args, **kwargs: {}
        )

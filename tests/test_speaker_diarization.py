import numpy as np

from conversation_analysis import TranscriptSegment
from speaker_diarization import (
    SpeakerTurn,
    align_transcript_to_speakers,
    build_speaker_aware_report,
    diarize_waveform,
)


def test_transcript_segments_align_by_maximum_overlap():
    turns = (
        SpeakerTurn(0, 5, "Speaker A", 0.9),
        SpeakerTurn(5, 10, "Speaker B", 0.8),
    )
    segments = (
        TranscriptSegment(1, 4, "Thank you"),
        TranscriptSegment(6, 9, "Send money immediately"),
    )
    aligned = align_transcript_to_speakers(segments, turns)
    assert [segment.speaker_label for segment in aligned] == ["Speaker A", "Speaker B"]


def test_speaker_report_keeps_risk_separate_by_speaker():
    turns = (
        SpeakerTurn(0, 5, "Speaker A", 0.9),
        SpeakerTurn(5, 15, "Speaker B", 0.9),
    )
    segments = (
        TranscriptSegment(1, 4, "Thank you for calling"),
        TranscriptSegment(6, 12, "Urgent, send money and share the OTP"),
    )
    report = build_speaker_aware_report(segments, turns)
    summaries = {summary.speaker_label: summary for summary in report.speakers}
    assert summaries["Speaker A"].conversation.risk_level == "Low"
    assert summaries["Speaker B"].conversation.risk_level == "High"


def test_lightweight_diarizer_detects_two_acoustically_distinct_regions():
    sample_rate = 16_000
    time = np.arange(4 * sample_rate) / sample_rate
    waveform = np.concatenate(
        [
            0.2 * np.sin(2 * np.pi * 180 * time),
            0.2 * np.sin(2 * np.pi * 880 * time),
        ]
    ).astype(np.float32)
    turns = diarize_waveform(waveform, sample_rate)
    assert {turn.speaker_label for turn in turns} == {"Speaker A", "Speaker B"}
    assert all(0.5 <= turn.confidence <= 0.99 for turn in turns)

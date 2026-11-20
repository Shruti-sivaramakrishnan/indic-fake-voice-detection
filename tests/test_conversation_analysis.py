import pytest

from conversation_analysis import (
    TranscriptSegment,
    analyze_transcript,
    parse_transcript_csv,
)


def test_parse_and_analyze_multilingual_cues():
    payload = (
        "start_seconds,end_seconds,text\n"
        "0,4,Thank you for taking my call\n"
        "10,14,यह urgent है अभी otp बताओ\n"
        "20,24,பணம் அனுப்பு யாரிடமும் சொல்லாதே\n"
    ).encode()
    segments = parse_transcript_csv(payload, call_duration_seconds=30)
    report = analyze_transcript(segments)

    assert report.segments == 3
    assert report.sentiment == "positive"
    assert report.category_counts["urgency"] >= 2
    assert report.category_counts["credential_or_otp"] == 1
    assert report.category_counts["money_or_payment"] == 1
    assert report.category_counts["secrecy"] == 1
    assert report.first_elevated_cue_time == 10


def test_sentiment_can_be_mixed_without_claiming_intent():
    report = analyze_transcript(
        [TranscriptSegment(0, 3, "Thanks, but I am worried and afraid")]
    )
    assert report.sentiment == "mixed"
    assert report.cue_hits == ()


@pytest.mark.parametrize(
    "payload,message",
    [
        (b"text\nhello\n", "needs"),
        (b"start_seconds,end_seconds,text\n2,1,bad\n", "invalid"),
        (b"start_seconds,end_seconds,text\n0,20,late\n", "beyond"),
    ],
)
def test_bad_transcript_is_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_transcript_csv(payload, call_duration_seconds=10)

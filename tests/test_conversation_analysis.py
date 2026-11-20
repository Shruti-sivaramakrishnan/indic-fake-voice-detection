import pytest

from conversation_analysis import (
    TranscriptSegment,
    _normalize,
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
    assert report.tone == "positive"
    assert report.risk_level == "High"
    assert report.category_counts["urgency"] == 1
    assert report.category_counts["credential_or_otp"] == 1
    assert report.category_counts["money_or_payment"] == 1
    assert report.category_counts["secrecy"] == 1
    assert report.first_elevated_cue_time == 10


def test_sentiment_can_be_mixed_without_claiming_intent():
    report = analyze_transcript(
        [TranscriptSegment(0, 3, "Thanks, but I am worried and afraid")]
    )
    assert report.tone == "mixed"
    assert report.risk_level == "Low"
    assert report.cue_hits == ()


def test_indic_normalization_preserves_vowel_signs_and_virama():
    assert _normalize("पैसे") == "पैसे"
    assert _normalize("நன்றி") == "நன்றி"
    assert _normalize("ಧನ್ಯವಾದ") == "ಧನ್ಯವಾದ"


def test_hindi_payment_variant_is_counted_once_without_false_matches():
    payment = analyze_transcript([TranscriptSegment(0, 3, "पैसे भेजो")])
    assert payment.category_counts == {"money_or_payment": 1}
    assert len(payment.cue_hits) == 1

    for benign_text in ("पूसा भाजी", "पास भेजो"):
        benign = analyze_transcript([TranscriptSegment(0, 3, benign_text)])
        assert benign.category_counts == {}
        assert benign.risk_level == "Low"


def test_generic_hindi_now_is_not_an_urgency_cue():
    report = analyze_transcript([TranscriptSegment(0, 3, "अभी नहीं")])
    assert report.category_counts == {}


def test_sample_transcript_has_positive_tone_but_high_risk():
    from pathlib import Path

    payload = Path("docs/sample_call_transcript.csv").read_bytes()
    report = analyze_transcript(parse_transcript_csv(payload, 180))

    assert report.tone == "positive"
    assert report.risk_level == "High"
    assert set(report.category_counts) >= {
        "urgency",
        "secrecy",
        "credential_or_otp",
        "money_or_payment",
    }


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

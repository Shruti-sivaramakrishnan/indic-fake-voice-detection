from types import SimpleNamespace

from pypdf import PdfReader

from evidence_report import generate_analysis_report


def test_analysis_report_is_one_page_and_uses_safe_language() -> None:
    result = SimpleNamespace(
        verdict="fake",
        confidence=0.99,
        raw_score=4.2,
        audio_hash="a" * 64,
        metadata={
            "p_fake_calibrated": 0.97,
            "duration_seconds": 4.2,
            "sample_rate_in": 16000,
            "frontend": "frontend",
            "checkpoint": "best_model.pt",
            "checkpoint_epoch": 11,
            "temperature": 1.26,
            "decision_policy_id": "legacy-symmetric-abstention-v1",
            "audio_quality": {
                "active_speech_seconds_estimate": 3.5,
                "rms": 0.05,
                "peak": 0.7,
                "quality_warning": None,
            },
        },
    )

    pdf = generate_analysis_report(result, "sample.wav")
    reader = PdfReader(__import__("io").BytesIO(pdf))
    text = " ".join(
        "\n".join(page.extract_text() or "" for page in reader.pages).split()
    )

    assert len(reader.pages) == 1
    assert "Automated Analysis Report" in text
    assert "not forensic evidence" in text
    assert "not a guaranteed real-world probability" in text
    assert "Evidence Report" not in text

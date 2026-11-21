from pathlib import Path

import pytest

from conversation_validation import (
    character_error_rate,
    evaluate_cases,
    load_validation_cases,
    word_error_rate,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "conversation_validation"
    / "labelled_regression_cases.json"
)


def test_checked_in_multilingual_fixture_is_green():
    results = evaluate_cases(load_validation_cases(FIXTURE))
    assert results["overall"]["cases"] == 16
    assert set(results["by_language"]) == {"English", "Hindi", "Tamil", "Kannada"}
    assert results["overall"]["risk_accuracy"] == 1.0
    assert results["overall"]["category_f1"] == 1.0
    assert results["overall"]["false_positive_rate"] == 0.0


def test_word_and_character_error_rates_are_unicode_safe():
    assert word_error_rate("पैसे भेजो", "पैसे भेजो") == 0.0
    assert word_error_rate("one two three", "one three") == pytest.approx(1 / 3)
    assert character_error_rate("நன்றி", "நன்றி") == 0.0

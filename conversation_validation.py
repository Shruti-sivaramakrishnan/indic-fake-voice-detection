"""Reproducible multilingual validation for conversation risk screening."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from conversation_analysis import TranscriptSegment, _normalize, analyze_transcript


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    language: str
    expected_risk: str
    expected_categories: tuple[str, ...]
    segments: tuple[TranscriptSegment, ...]
    reference_transcript: str | None = None
    hypothesis_transcript: str | None = None


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, ref_item in enumerate(reference, start=1):
        current = [index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = _normalize(reference).split()
    hypothesis_words = _normalize(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _edit_distance(reference_words, hypothesis_words) / len(reference_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_characters = list(_normalize(reference).replace(" ", ""))
    hypothesis_characters = list(_normalize(hypothesis).replace(" ", ""))
    if not reference_characters:
        return 0.0 if not hypothesis_characters else 1.0
    return _edit_distance(reference_characters, hypothesis_characters) / len(reference_characters)


def load_validation_cases(path: str | Path) -> tuple[ValidationCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for item in payload:
        segments = tuple(
            TranscriptSegment(
                float(segment["start_seconds"]),
                float(segment["end_seconds"]),
                str(segment["text"]),
            )
            for segment in item["segments"]
        )
        cases.append(
            ValidationCase(
                case_id=str(item["case_id"]),
                language=str(item["language"]),
                expected_risk=str(item["expected_risk"]),
                expected_categories=tuple(sorted(item.get("expected_categories", []))),
                segments=segments,
                reference_transcript=item.get("reference_transcript"),
                hypothesis_transcript=item.get("hypothesis_transcript"),
            )
        )
    if not cases:
        raise ValueError("Validation fixture contains no cases.")
    return tuple(cases)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _aggregate(rows: Iterable[dict]) -> dict:
    selected = list(rows)
    risk_correct = sum(row["risk_correct"] for row in selected)
    tp = sum(row["category_tp"] for row in selected)
    fp = sum(row["category_fp"] for row in selected)
    fn = sum(row["category_fn"] for row in selected)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    benign = [row for row in selected if row["expected_risk"] == "Low"]
    false_positive_cases = sum(row["predicted_risk"] != "Low" for row in benign)
    wer_values = [row["wer"] for row in selected if row["wer"] is not None]
    cer_values = [row["cer"] for row in selected if row["cer"] is not None]
    return {
        "cases": len(selected),
        "risk_accuracy": _safe_ratio(risk_correct, len(selected)),
        "category_precision": precision,
        "category_recall": recall,
        "category_f1": _safe_ratio(2 * precision * recall, precision + recall),
        "benign_cases": len(benign),
        "false_positive_rate": _safe_ratio(false_positive_cases, len(benign)),
        "mean_wer": sum(wer_values) / len(wer_values) if wer_values else None,
        "mean_cer": sum(cer_values) / len(cer_values) if cer_values else None,
    }


def evaluate_cases(cases: Iterable[ValidationCase]) -> dict:
    rows = []
    for case in cases:
        report = analyze_transcript(case.segments)
        expected = set(case.expected_categories)
        predicted = set(report.category_counts)
        has_asr_pair = (
            case.reference_transcript is not None
            and case.hypothesis_transcript is not None
        )
        rows.append(
            {
                "case_id": case.case_id,
                "language": case.language,
                "expected_risk": case.expected_risk,
                "predicted_risk": report.risk_level,
                "risk_correct": report.risk_level == case.expected_risk,
                "expected_categories": sorted(expected),
                "predicted_categories": sorted(predicted),
                "category_tp": len(expected & predicted),
                "category_fp": len(predicted - expected),
                "category_fn": len(expected - predicted),
                "wer": (
                    word_error_rate(case.reference_transcript or "", case.hypothesis_transcript or "")
                    if has_asr_pair
                    else None
                ),
                "cer": (
                    character_error_rate(case.reference_transcript or "", case.hypothesis_transcript or "")
                    if has_asr_pair
                    else None
                ),
            }
        )
    languages = sorted({row["language"] for row in rows})
    return {
        "suite_scope": "labelled regression and acceptance fixture; not a population benchmark",
        "overall": _aggregate(rows),
        "by_language": {
            language: _aggregate(row for row in rows if row["language"] == language)
            for language in languages
        },
        "cases": rows,
    }


def write_results(results: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

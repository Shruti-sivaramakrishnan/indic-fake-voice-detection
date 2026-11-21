"""Run the checked-in multilingual conversation regression fixture."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from conversation_validation import evaluate_cases, load_validation_cases, write_results  # noqa: E402


FIXTURE = REPO_ROOT / "evaluation" / "conversation_validation" / "labelled_regression_cases.json"
OUTPUT_DIR = FIXTURE.parent


def main() -> int:
    results = evaluate_cases(load_validation_cases(FIXTURE))
    write_results(results, OUTPUT_DIR / "validation_summary.json")
    with (OUTPUT_DIR / "per_language_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "language", "cases", "risk_accuracy", "category_precision",
            "category_recall", "category_f1", "benign_cases", "false_positive_rate",
            "mean_wer", "mean_cer",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for language, metrics in results["by_language"].items():
            writer.writerow({"language": language, **metrics})
    overall = results["overall"]
    print(
        f"Validated {overall['cases']} cases: risk accuracy={overall['risk_accuracy']:.3f}, "
        f"category F1={overall['category_f1']:.3f}, FPR={overall['false_positive_rate']:.3f}"
    )
    return 0 if overall["risk_accuracy"] == 1.0 and overall["category_f1"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

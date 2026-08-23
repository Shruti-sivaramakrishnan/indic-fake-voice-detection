"""Audit historical prediction CSVs for fake/bonafide threshold sign errors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def confusion(labels: List[int], logits: List[float], threshold: float) -> Dict[str, float]:
    predictions = [int(logit > threshold) for logit in logits]
    tn = sum(label == 0 and pred == 0 for label, pred in zip(labels, predictions))
    fp = sum(label == 0 and pred == 1 for label, pred in zip(labels, predictions))
    fn = sum(label == 1 and pred == 0 for label, pred in zip(labels, predictions))
    tp = sum(label == 1 and pred == 1 for label, pred in zip(labels, predictions))
    return {
        "accuracy": (tn + tp) / len(labels),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def audit_file(path: Path, bonafide_threshold: float) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = [int(row["true_label"]) for row in rows]
    logits = [float(row["logit"]) for row in rows]
    return {
        "file": str(path.relative_to(REPO_ROOT)),
        "n": len(rows),
        "operating_points": {
            "raw_logit_zero": {
                "fake_logit_threshold": 0.0,
                **confusion(labels, logits, 0.0),
            },
            "historical_wrong_sign": {
                "fake_logit_threshold": bonafide_threshold,
                "warning": "Bonafide-score threshold was used without negating it.",
                **confusion(labels, logits, bonafide_threshold),
            },
            "validation_eer_correct_orientation": {
                "fake_logit_threshold": -bonafide_threshold,
                "source": "validation EER; not selected on test accuracy",
                **confusion(labels, logits, -bonafide_threshold),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path, default=REPO_ROOT / "checkpoints" / "step6_summary.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "evaluation" / "threshold_orientation_audit.json",
    )
    parser.add_argument(
        "predictions",
        type=Path,
        nargs="*",
        default=[
            REPO_ROOT / "evaluation" / "clean" / "frozen_test_predictions_with_language.csv",
            REPO_ROOT / "evaluation" / "robustness" / "opus_16kbps_predictions.csv",
            REPO_ROOT / "evaluation" / "robustness" / "musan_10db_predictions.csv",
        ],
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    bonafide_threshold = float(
        summary.get(
            "val_eer_threshold_bonafide_score",
            summary["val_eer_threshold"],
        )
    )
    payload = {
        "finding": (
            "Historical robustness classification artifacts used the validation "
            "bonafide-score threshold as a fake-logit threshold without negating it."
        ),
        "validation_eer_threshold_bonafide_score": bonafide_threshold,
        "validation_eer_threshold_fake_logit": -bonafide_threshold,
        "selection_warning": (
            "The corrected validation threshold is reported for consistency; do "
            "not choose among these operating points using test accuracy."
        ),
        "datasets": [audit_file(path.resolve(), bonafide_threshold) for path in args.predictions],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

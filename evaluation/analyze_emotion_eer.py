"""Baseline per-emotion EER analysis for frozen Voice Guard predictions.

This utility analyses an existing prediction CSV; it never runs inference and
never changes model selection or thresholds. Results from the historical v1
split are diagnostic only because that split is known to have group overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EMOTION_TOKENS = {
    "ANGER": "Anger",
    "ANGRY": "Anger",
    "DISGUST": "Disgust",
    "FEAR": "Fear",
    "FEARFUL": "Fear",
    "HAPPY": "Happy",
    "HAPPINESS": "Happy",
    "SAD": "Sad",
    "SADNESS": "Sad",
    "SURPRISE": "Surprise",
    "SURPRISED": "Surprise",
}


def emotion_from_clip_id(clip_id: str) -> str | None:
    """Return an explicit emotion token embedded in a clip ID, if present."""
    normalized = clip_id.upper().replace("-", "_")
    for token in normalized.split("_"):
        if token in EMOTION_TOKENS:
            return EMOTION_TOKENS[token]
    return None


def equal_error_rate(labels: np.ndarray, fake_scores: np.ndarray) -> float:
    """Compute EER for labels 0=real, 1=fake and scores increasing with fake."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(fake_scores, dtype=np.float64)
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("labels and fake_scores must be non-empty and aligned")
    if not np.isfinite(scores).all():
        raise ValueError("fake_scores contain non-finite values")
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("EER requires both real and fake examples")

    thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    fpr = []
    fnr = []
    for threshold in thresholds:
        predicted_fake = scores >= threshold
        fpr.append(np.count_nonzero(predicted_fake & (labels == 0)) / negatives)
        fnr.append(np.count_nonzero(~predicted_fake & (labels == 1)) / positives)
    fpr_array = np.asarray(fpr)
    fnr_array = np.asarray(fnr)
    index = int(np.argmin(np.abs(fpr_array - fnr_array)))
    return float((fpr_array[index] + fnr_array[index]) / 2.0)


def bootstrap_eer_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a stratified percentile 95 percent confidence interval."""
    rng = np.random.default_rng(seed)
    real_indices = np.flatnonzero(labels == 0)
    fake_indices = np.flatnonzero(labels == 1)
    estimates = []
    for _ in range(iterations):
        sampled = np.r_[
            rng.choice(real_indices, size=real_indices.size, replace=True),
            rng.choice(fake_indices, size=fake_indices.size, replace=True),
        ]
        estimates.append(equal_error_rate(labels[sampled], scores[sampled]))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def analyse_predictions(
    input_csv: Path,
    bootstrap_iterations: int = 1000,
) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    total = 0
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"clip_id", "true_label", "logit"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Prediction CSV must include {sorted(required)}")
        for row in reader:
            total += 1
            emotion = emotion_from_clip_id(row["clip_id"])
            if emotion is not None:
                grouped[emotion].append(row)

    results = []
    for offset, emotion in enumerate(sorted(grouped)):
        rows = grouped[emotion]
        labels = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int8)
        scores = np.asarray([float(row["logit"]) for row in rows], dtype=np.float64)
        real_count = int(np.count_nonzero(labels == 0))
        fake_count = int(np.count_nonzero(labels == 1))
        if real_count == 0 or fake_count == 0:
            continue
        eer = equal_error_rate(labels, scores)
        ci_low, ci_high = bootstrap_eer_interval(
            labels,
            scores,
            iterations=bootstrap_iterations,
            seed=42 + offset,
        )
        results.append(
            {
                "emotion": emotion,
                "clips": len(rows),
                "real_clips": real_count,
                "fake_clips": fake_count,
                "eer": eer,
                "eer_ci95_low": ci_low,
                "eer_ci95_high": ci_high,
            }
        )

    summary = {
        "input_csv": str(input_csv),
        "total_prediction_rows": total,
        "emotion_labeled_rows": sum(len(rows) for rows in grouped.values()),
        "reportable_emotions": len(results),
        "bootstrap_iterations": bootstrap_iterations,
        "scope_warning": (
            "Diagnostic v1 result only. The historical split has known group overlap; "
            "repeat on the locked group-safe v2 test set before publication."
        ),
    }
    return results, summary


def write_outputs(results: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_emotion_eer.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(results[0]) if results else ["emotion"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    payload = {"summary": summary, "results": results}
    (output_dir / "per_emotion_eer.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Baseline per emotion EER",
        "",
        summary["scope_warning"],
        "",
        "| Emotion | Clips | Real | Fake | EER | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['emotion']} | {row['clips']} | {row['real_clips']} | "
            f"{row['fake_clips']} | {row['eer']:.2%} | "
            f"{row['eer_ci95_low']:.2%}-{row['eer_ci95_high']:.2%} |"
        )
    (output_dir / "PER_EMOTION_EER.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/clean/frozen_test_predictions_with_language.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/emotion"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    results, summary = analyse_predictions(args.input, args.bootstrap_iterations)
    write_outputs(results, summary, args.output_dir)
    print(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()

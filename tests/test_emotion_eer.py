import csv

import numpy as np

from evaluation.analyze_emotion_eer import (
    analyse_predictions,
    emotion_from_clip_id,
    equal_error_rate,
)


def test_emotion_is_read_only_from_explicit_id_token():
    assert emotion_from_clip_id("BEN_F_SAD_00197") == "Sad"
    assert emotion_from_clip_id("DOI-M-ANGER-00001") == "Anger"
    assert emotion_from_clip_id("train_gujaratimale_01293") is None
    assert emotion_from_clip_id("BEN_M_BOOK_00694") is None


def test_eer_is_zero_for_perfect_scores():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([-2.0, -1.0, 1.0, 2.0])
    assert equal_error_rate(labels, scores) == 0.0


def test_analysis_excludes_unlabeled_rows(tmp_path):
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip_id", "true_label", "logit"]
        )
        writer.writeheader()
        writer.writerows([
            {"clip_id": "BEN_F_SAD_1", "true_label": 0, "logit": -1.0},
            {"clip_id": "BEN_F_SAD_2", "true_label": 1, "logit": 1.0},
            {"clip_id": "train_gujarati_3", "true_label": 0, "logit": -2.0},
        ])

    results, summary = analyse_predictions(path, bootstrap_iterations=5)
    assert len(results) == 1
    assert results[0]["emotion"] == "Sad"
    assert results[0]["eer"] == 0.0
    assert summary["total_prediction_rows"] == 3
    assert summary["emotion_labeled_rows"] == 2

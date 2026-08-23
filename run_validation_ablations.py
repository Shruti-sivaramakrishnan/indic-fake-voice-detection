"""Compare v2 model heads and robustness augmentation on validation only.

This script never evaluates the test split. It reuses one frozen-feature cache,
stores each candidate in a separate checkpoint directory, and writes a ranked
validation summary. Only the selected configuration should proceed to a single
v2 test evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict

import model_pipeline as mp


def candidate_configs(base: mp.TrainConfig) -> Dict[str, mp.TrainConfig]:
    return {
        "gat_clean": replace(base, head_type="gat"),
        "gat_separate_norms_clean": replace(
            base, head_type="gat", separate_layer_norms=True
        ),
        "attention_pool_clean": replace(base, head_type="attention_pool"),
        "mean_pool_clean": replace(base, head_type="mean_pool"),
        # Starting proposal, not a claimed optimum. Validation decides whether it
        # helps; the frozen test set is never inspected during this comparison.
        "gat_feature_robust": replace(
            base,
            head_type="gat",
            separate_layer_norms=True,
            feature_noise_std=0.03,
            time_mask_probability=0.5,
            time_mask_max_frames=20,
            feature_channel_dropout_probability=0.02,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-file",
        default="data_pipeline/split_ids_v2.json",
        help="Group-disjoint v2 split produced by build_group_safe_split.py.",
    )
    parser.add_argument(
        "--metadata-file",
        default=mp.METADATA_FILE,
        help="Metadata containing labels/languages for every v2 clip ID.",
    )
    parser.add_argument("--cache-dir", default="feature_cache_v2")
    parser.add_argument("--checkpoint-root", default="experiments/v2_ablations")
    parser.add_argument(
        "--output", default="evaluation/validation_ablation_results.json"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--max-clips-per-split", type=int, default=None)
    args = parser.parse_args()

    split_path = Path(args.split_file)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing {split_path}. Build and audit a group-disjoint v2 split first."
        )

    base = mp.TrainConfig(num_epochs=args.epochs, batch_size=args.batch_size)
    summaries = {}
    for name, config in candidate_configs(base).items():
        print("\n" + "=" * 72)
        print(f"VALIDATION ABLATION: {name}")
        print("=" * 72)
        checkpoint_dir = str(Path(args.checkpoint_root) / name)
        summary = mp.run_step6_training(
            config=config,
            split_file=str(split_path),
            metadata_file=args.metadata_file,
            cache_dir=args.cache_dir,
            checkpoint_dir=checkpoint_dir,
            extract_batch_size=args.extract_batch_size,
            max_clips_per_split=args.max_clips_per_split,
        )
        summaries[name] = {
            "config": summary["config"],
            "val_eer": summary["val_eer"],
            "ece_before": summary["ece_before"],
            "ece_after": summary["ece_after"],
            "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
            "split_sha256": summary["split_sha256"],
        }

    ranking = sorted(summaries, key=lambda name: summaries[name]["val_eer"])
    payload = {
        "selection_scope": "validation_only",
        "test_split_touched": False,
        "best_candidate_by_val_eer": ranking[0],
        "ranking": ranking,
        "candidates": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    print(f"Best validation candidate: {ranking[0]}")
    print("Review robustness results before the one-time v2 test evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

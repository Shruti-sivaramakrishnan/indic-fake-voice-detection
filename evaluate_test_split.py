"""
Evaluate the trained model on the frozen TEST split (3,111 clips) — the split
Step 6 never touched during training or model selection.

Run this exactly once, after every Step 6 training/hyperparameter decision is
final. Its output is the first real, reportable test-set number for the
project. Re-running it after going back and changing something in Step 6 turns
the test set into a second validation set and invalidates that property.

Usage (Colab/Kaggle, after Step 6 has produced checkpoints/best_model.pt):

    !python evaluate_test_split.py \\
        --checkpoint-dir /content/drive/MyDrive/indic-fake-voice-detection/checkpoints \\
        --cache-dir /content/drive/MyDrive/indic-fake-voice-detection/feature_cache

or from a notebook cell:

    import evaluate_test_split as t
    result = t.main(
        checkpoint_dir="/content/drive/MyDrive/indic-fake-voice-detection/checkpoints",
        cache_dir="/content/drive/MyDrive/indic-fake-voice-detection/feature_cache",
    )

Feature extraction for the test split reuses the same streaming/batching/resume/
failure-handling machinery as Step 6 (build_feature_caches_streaming via
evaluate_dataset -> ensure_features_cached_for_ids), cached under the "test"
split name in the same feature_cache directory — separate from train/validation,
so it survives independently and can resume after a disconnect the same way
Step 6 does.

On a full (non-rehearsal) run this writes:
    docs/TEST_SET_RESULTS.md              pooled + per-language metrics
    checkpoints/per_language_test_metrics.json (benchmark only)
    docs/error_analysis/fp_clips.csv      real clips misclassified as fake
    docs/error_analysis/fn_clips.csv      fake clips misclassified as real

The FP/FN CSVs are the exact per-clip breakdown behind TEST_SET_RESULTS.md's
confusion matrix — same run, same cache, same model, so the two can never
silently disagree (write_error_analysis() asserts the row counts match the
confusion matrix cells before writing).
"""

import argparse
import hashlib
import sys
from typing import Optional

import model_pipeline as mp


def main(
    checkpoint_dir: str = mp.CHECKPOINT_DIR,
    cache_dir: str = mp.FEATURE_CACHE_DIR,
    repo_dir: str = mp.REPO_ROOT,
    split_file: str = mp.SPLIT_FILE,
    metadata_file: str = mp.METADATA_FILE,
    decision_threshold: Optional[float] = None,
    extract_batch_size: int = 8,
    min_language_support: int = 50,
    max_clips: Optional[int] = None,
) -> dict:
    print("=" * 64)
    print("TEST SET EVALUATION — frozen split, never touched by Step 6")
    print("=" * 64)
    print(
        "This should run exactly once per final model. If you're re-running "
        "this after changing something in Step 6, stop — that makes the test "
        "set a second validation set.\n"
    )

    print(f"device         : {mp.DEVICE}")
    print(f"checkpoint dir : {checkpoint_dir}")
    print(f"cache dir      : {cache_dir}\n")

    # --- guards, same as Step 6 --------------------------------------------
    print("[1/4] Verifying frozen split integrity...")
    splits = mp.load_frozen_split(split_file)
    mp.assert_split_integrity(
        splits,
        require_group_disjoint=True,
        split_file=split_file,
        metadata_file=metadata_file,
    )
    metadata = mp.load_verified_metadata(metadata_file)
    mp.assert_split_coverage(splits, metadata)
    print("  OK - groups are disjoint and every language/label is represented.\n")

    print("[2/4] Verifying frontend is frozen...")
    n_trainable = mp.verify_frontend_frozen()
    print(f"  frontend trainable params: {n_trainable}\n")

    # --- build the test clip_ids/labels/languages from frozen metadata -----
    print("[3/4] Loading frozen TEST split and verified metadata...")
    _, _, test_ids = mp.load_frozen_split_sets(split_file)
    label_lookup, language_lookup = mp.build_label_lookup(metadata)

    clip_ids = sorted(test_ids)   # deterministic order
    if max_clips is not None:
        clip_ids = clip_ids[:max_clips]
        print(f"  max_clips={max_clips} — REHEARSAL, not a real test-set run.")
    labels = [label_lookup[c] for c in clip_ids]

    n_real = sum(1 for v in labels if v == mp.LABEL_REAL)
    n_fake = sum(1 for v in labels if v == mp.LABEL_FAKE)
    print(f"  {len(clip_ids):,} test clips ({n_real:,} real, {n_fake:,} fake)\n")

    # --- evaluate ------------------------------------------------------------
    print("[4/4] Evaluating (builds/reuses the 'test' feature cache, then scores)...")
    result = mp.evaluate_dataset(
        clip_ids=clip_ids,
        labels=labels,
        checkpoint_dir=checkpoint_dir,
        cache_dir=cache_dir,
        cache_split_name="test",
        language_lookup=language_lookup,
        decision_threshold=decision_threshold,
        extract_batch_size=extract_batch_size,
        # Aggregate benchmark artifact only. It is intentionally never consumed
        # as an individual clip's confidence or risk estimate.
        write_language_eer_file=(max_clips is None),
        # Per-clip records feed the error-analysis CSVs below. Skipped on a
        # rehearsal since those results aren't written anywhere anyway.
        include_per_clip=(max_clips is None),
    )
    with open(split_file, "rb") as split_stream:
        result["split_sha256"] = hashlib.sha256(split_stream.read()).hexdigest()
    result["split_protocol"] = "v2-group-disjoint"

    # --- report ---------------------------------------------------------------
    pooled = result["pooled"]
    cm = pooled["confusion_matrix"]

    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"  evaluated      : {result['n_evaluated']:,} / {result['n_requested']:,}")
    if result["missing_clip_ids"]:
        print(f"  missing        : {len(result['missing_clip_ids']):,} "
              f"(decode failure or absent from stream)")

    eer_str = (
        f"{pooled['eer'] * 100:.2f}%" if pooled["eer"] is not None else "undefined"
    )
    print(f"\n  TEST EER       : {eer_str}")
    print(f"  accuracy       : {pooled['accuracy'] * 100:.2f}%")
    print(f"  precision      : {pooled['precision'] * 100:.2f}%  (positive=fake)")
    print(f"  recall         : {pooled['recall'] * 100:.2f}%")
    print(f"  f1             : {pooled['f1']:.4f}")
    print(f"  confusion      : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")

    print("\n  per-language EER:")
    for lang, entry in sorted(result["per_language"].items()):
        eer_s = f"{entry['eer'] * 100:.2f}%" if entry["eer"] is not None else "n/a"
        flag = " (LOW SUPPORT)" if entry["n"] < min_language_support else ""
        print(f"    {lang:<12} n={entry['n']:<5} EER={eer_s:<8}{flag}")

    if max_clips is None:
        print("\nWriting results...")
        results_path = mp.write_test_set_summary(
            result, repo_dir, min_language_support=min_language_support
        )
        print(f"  wrote {results_path}")

        error_paths = mp.write_error_analysis(result, repo_dir)
        print(
            f"  wrote {error_paths['fp_path']} ({error_paths['n_fp']} rows)\n"
            f"  wrote {error_paths['fn_path']} ({error_paths['n_fn']} rows)"
        )
        print(
            f"  (confusion matrix agrees: FP={cm['fp']}=={error_paths['n_fp']}, "
            f"FN={cm['fn']}=={error_paths['n_fn']})"
        )

        print("\nReview every generated artifact before any commit or push.")
        result["results_path"] = results_path
        result["error_analysis_paths"] = error_paths
    else:
        print("\nREHEARSAL run (--max-clips set) — no results file, "
              "no per-language benchmark JSON, no error-analysis CSVs written. "
              "Not reportable.")

    print("=" * 64)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=mp.CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", default=mp.FEATURE_CACHE_DIR)
    parser.add_argument("--repo-dir", default=mp.REPO_ROOT)
    parser.add_argument("--split-file", default=mp.SPLIT_FILE)
    parser.add_argument("--metadata-file", default=mp.METADATA_FILE)
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="P(fake) threshold. Defaults to the validation EER operating point.",
    )
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--min-language-support", type=int, default=50)
    parser.add_argument(
        "--max-clips", type=int, default=None,
        help="Cap clips for a cheap rehearsal. Skips writing results — not reportable.",
    )
    args = parser.parse_args()

    main(
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=args.cache_dir,
        repo_dir=args.repo_dir,
        split_file=args.split_file,
        metadata_file=args.metadata_file,
        decision_threshold=args.decision_threshold,
        extract_batch_size=args.extract_batch_size,
        min_language_support=args.min_language_support,
        max_clips=args.max_clips,
    )
    sys.exit(0)

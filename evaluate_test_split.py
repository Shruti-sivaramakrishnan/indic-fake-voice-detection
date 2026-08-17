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
"""

import argparse
import sys
from typing import Optional

import model_pipeline as mp


def main(
    checkpoint_dir: str = mp.CHECKPOINT_DIR,
    cache_dir: str = mp.FEATURE_CACHE_DIR,
    repo_dir: str = mp.REPO_ROOT,
    decision_threshold: float = 0.5,
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
    mp.assert_split_integrity()
    print("  OK — 31,102 clips, no duplicates, no cross-split overlap.\n")

    print("[2/4] Verifying frontend is frozen...")
    n_trainable = mp.verify_frontend_frozen()
    print(f"  frontend trainable params: {n_trainable}\n")

    # --- build the test clip_ids/labels/languages from frozen metadata -----
    print("[3/4] Loading frozen TEST split and verified metadata...")
    _, _, test_ids = mp.load_frozen_split_sets()
    label_lookup, language_lookup = mp.build_label_lookup()

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
        # This run is exactly the intended source of truth for per-language EER
        # -- write it so analyze_bytes() stops returning language_eer=None for
        # every known language.
        write_language_eer_file=(max_clips is None),
    )

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
        print("\nNext action — commit:")
        print(f"  cd {repo_dir}")
        print("  git add docs/TEST_SET_RESULTS.md")
        print(f'  git commit -m "Test set results: EER {eer_str}"')
        print("  git push -u origin claude/ml-repo-structure-review-fpox8u")
        print(
            "\n  (git identity for this repo is Shruti <shruti@example.com> — "
            "verify with `git config --local --list | grep user` before "
            "committing if you're in a fresh environment.)"
        )
        result["results_path"] = results_path
    else:
        print("\nREHEARSAL run (--max-clips set) — no results file written, "
              "no per_language_eer.json written. Not reportable.")

    print("=" * 64)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=mp.CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", default=mp.FEATURE_CACHE_DIR)
    parser.add_argument("--repo-dir", default=mp.REPO_ROOT)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
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
        decision_threshold=args.decision_threshold,
        extract_batch_size=args.extract_batch_size,
        min_language_support=args.min_language_support,
        max_clips=args.max_clips,
    )
    sys.exit(0)

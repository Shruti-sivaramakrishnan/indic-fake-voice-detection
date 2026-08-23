# Voice Guard v2 training and evaluation protocol

This procedure is the path to a more trustworthy accuracy result. It prevents prompt leakage, rebuilds feature caches, compares model heads on validation only, and keeps the v2 test set untouched until one candidate is locked.

## What the protocol can claim

The current upstream dataset exposes clip ID, language, transcript, label, and audio. It does not expose stable speaker, recording session, source corpus, or generator lineage. The included workflow can therefore establish a **prompt-group-disjoint** evaluation. It cannot establish speaker-disjoint, source-disjoint, or unseen-generator performance without new metadata.

## Run on Kaggle

1. Open `kaggle/voice_guard_v2_training.ipynb` in Kaggle and enable a GPU plus internet access.
2. Leave `VOICE_GUARD_TRAIN_MODE=smoke` for the first run. This caps each split and uses two epochs to verify metadata export, split construction, cache creation, training, checkpointing, and validation output.
3. Confirm that `data_pipeline/split_ids_v2.audit.json` reports zero group overlap, every language/label stratum in every split, and a SHA-256 matching the split file.
4. Change `VOICE_GUARD_TRAIN_MODE` to `full`, restart the notebook session, and run from the top. Full mode uses fresh `feature_cache_v2` and `experiments/v2_ablations` paths.
5. Compare the clean GAT, separate-normalisation GAT, attention-pooling, mean-pooling, and feature-robust GAT candidates using validation EER and calibration. Add the locked robustness matrix before the final choice; do not inspect v2 test labels or scores during this decision.
6. Record the winning checkpoint path in `VOICE_GUARD_LOCKED_CHECKPOINT`. Set `VOICE_GUARD_RUN_FINAL_TEST=1` only after every model, augmentation, and threshold choice is final.
7. Run the gated final-test cell once. Archive the exact Git SHA, split and audit SHA values, checkpoint, configuration, validation summary, and final metrics together.

## Acceptance checklist

- Metadata export contains the expected 31,102 rows and no hidden `-1` labels.
- The v2 split audit is present, protocol version is `v2-group-disjoint`, overlap count is zero, and all language/label strata are represented.
- No v1 cache or degraded-audio manifest is reused.
- Candidate selection says `selection_scope: validation_only` and `test_split_touched: false`.
- The final checkpoint was selected before enabling the test cell.
- Clean metrics are accompanied by codec/noise/channel tests; repeated seeds and bootstrap confidence intervals are added before publication-quality claims.

## Promoting a v2 checkpoint to the demo

Do not overwrite `checkpoints/best_model.pt` merely because a run completed. First verify that the v2 model improves the locked validation criteria and does not regress important robustness slices. Then copy the selected checkpoint and its matching summary into `checkpoints/`, run the repository tests and the local Streamlit smoke test, and create a separate reviewed commit. Until that happens, the demo intentionally remains on the historical v1 checkpoint and must be described that way.

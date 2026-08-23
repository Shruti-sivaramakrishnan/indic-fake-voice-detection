# Test Set Results — Model Pipeline

> **Historical v1 result - not a current generalization claim.** Exact clip IDs are disjoint, but the repaired leakage audit finds source-family proxy overlap across every split pair: 24,396/24,881 train clips and 3,090/3,111 test clips belong to families present on both sides. This v1 test set has also been observed and must not be used for further model, threshold, or augmentation selection. Future reportable results require the group-disjoint v2 protocol.

> The accuracy/confusion matrix below used `P(fake) > 0.5`. It did not use the validation EER threshold. Historical robustness artifacts also applied the saved bonafide-score threshold with the wrong sign in fake-logit space; regenerate those artifacts for v2 with the repaired evaluator.

_Generated 2026-08-17 09:47:27 by `evaluate_test_split.py`, via `model_pipeline.evaluate_dataset()`._

**This is the frozen TEST split — the one Step 6 never touched during training or model selection. It should be evaluated exactly once, after every training/hyperparameter decision is final. If you are looking at this file wondering whether to go change something in Step 6 and re-run this — don't; that turns the test set into a second validation set.**

## Headline

- **Test EER: 1.03%**
- Accuracy: 98.39%
- Precision: 99.67% (positive class = fake)
- Recall: 97.10%
- F1: 0.9837
- Decision threshold: P(fake) > 0.5 (used for accuracy/precision/recall/f1/confusion matrix only — EER itself is threshold-independent)

## Confusion matrix

| | predicted real | predicted fake |
|---|---|---|
| **actual real** | 1553 (TN) | 5 (FP) |
| **actual fake** | 45 (FN) | 1508 (TP) |

## Coverage

- Requested: 3,111
- Evaluated: 3,111
- Missing (decode failure or absent from stream): 0

## Per-language test EER (aggregate benchmark only)

These values describe buckets in this historical test split. They are not an individual upload's confidence or risk estimate and are no longer displayed in the UI/PDF.

| language | n | n_real | n_fake | EER | support |
|---|---|---|---|---|---|
| Assamese | 200 | 100 | 100 | 0.00% | ok |
| Bengali | 180 | 90 | 90 | 0.00% | ok |
| Bodo | 200 | 101 | 99 | 0.00% | ok |
| Dogri | 200 | 100 | 100 | 0.00% | ok |
| English | 200 | 100 | 100 | 0.00% | ok |
| Gujarati | 156 | 78 | 78 | 0.00% | ok |
| Hindi | 200 | 100 | 100 | 0.00% | ok |
| Kannada | 200 | 100 | 100 | 0.00% | ok |
| Malayalam | 201 | 101 | 100 | 0.50% | ok |
| Manipuri | 200 | 100 | 100 | 1.00% | ok |
| Marathi | 201 | 101 | 100 | 0.00% | ok |
| Nepali | 200 | 100 | 100 | 2.50% | ok |
| Odia | 182 | 91 | 91 | 2.20% | ok |
| Sanskrit | 200 | 100 | 100 | 0.00% | ok |
| Tamil | 191 | 96 | 95 | 0.00% | ok |
| Telugu | 200 | 100 | 100 | 0.00% | ok |

## Calibration

- Temperature: 1.2600
- Calibrated (vs. temperature=1.0 fallback): True
- Checkpoint: `best_model.pt` (epoch 11)

## Caveats

- 1 seed, not 3 — compressed 15-day timeline. This EER has no cross-seed variance estimate.
- Speaker-disjointness across splits cannot be verified from this repo (no speaker field in the metadata).
- Features cached in float16; not re-verified against fp32 on this run.

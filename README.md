# Voice Guard - Indic fake voice detection

Voice Guard is a research/demo pipeline for detecting synthetic speech across 16 Indian languages. It uses a frozen `facebook/wav2vec2-xls-r-300m` frontend and a trainable graph-attention classifier head, exposed through a Streamlit upload UI.

The head is **AASIST-inspired, not an implementation of the full AASIST architecture**. The current checkpoint is a historical v1 model and must not be treated as production-validated.

## Current status

- The upload path now rejects empty, silent, too-short, too-long, non-finite, and oversized audio before model inference.
- The UI returns `Likely genuine`, `Inconclusive`, or `Likely synthetic`, and labels its number as a model score rather than proof.
- Results and PDF downloads are bound to the exact SHA-256 hash of the currently selected audio.
- Automatic language identification is disabled. The removed SpeechBrain path did not correctly parse/resample all inputs and did not cover all 16 target languages.
- Evaluation threshold orientation is explicit in both bonafide-score and fake-logit space.
- New training/evaluation refuses the historical v1 split because source-family proxy groups cross train, validation, and test boundaries.

Historical v1 metrics remain useful for regression comparison, but they are optimistic for generalization: exact clip IDs are disjoint while approximately 98-99% of clips belong to ID families represented across split boundaries. See [docs/TEST_SET_RESULTS.md](docs/TEST_SET_RESULTS.md) and run the leakage audit below.

## Run the demo locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The demo expects:

- `checkpoints/best_model.pt`
- `checkpoints/step6_summary.json`
- network/disk access for the first XLS-R frontend download

Accepted formats are WAV, MP3, FLAC, OGG, and M4A. The public input policy is 25 MB maximum, 1-30 seconds, with at least about half a second of audible speech.

### Transmission robustness comparison

After a normal analysis, the demo can create three in-memory variants of the
same upload and score them with the same checkpoint:

- Opus at 16 kbps in VoIP mode;
- G.711 A-law at 8 kHz with telephone-band filtering; and
- 8 kHz telephone-band linear PCM.

The comparison reports each AI-likelihood score, its change from the original,
and whether the verdict changed. FFmpeg must be installed and available on
`PATH`; transformed audio is not saved to disk. This is a diagnostic of the
current model under simulated transmission conditions, not evidence that the
model has been retrained or proven codec-robust.

## Test the dependency-light safety layer

```bash
python -m pip install -r requirements-training.txt
pytest -q tests
python data_pipeline/leakage_audit.py
```

The last command is expected to **fail for `data_pipeline/split_ids.json`**. That failure documents the known v1 group overlap. Use `--allow-known-v1-group-overlap` only to inspect historical lineage; it is not approval for another reportable training run.

## Build the v2 data protocol

Preferred metadata contains real `speaker_id`, `source_id`, `session_id`, generator, and normalized prompt/transcript fields. Keep every relevant group in exactly one split.

```bash
python data_pipeline/export_hf_metadata.py \
  --output data_pipeline/metadata_v2.jsonl

python data_pipeline/build_group_safe_split.py \
  data_pipeline/metadata_v2.jsonl \
  --group-columns prompt_id \
  --output data_pipeline/split_ids_v2.json

python data_pipeline/leakage_audit.py \
  --split-file data_pipeline/split_ids_v2.json
```

The official upstream dataset currently exposes transcript text but not stable speaker, session, source, or generator identifiers. The exporter therefore derives a normalized, language-scoped prompt hash without downloading audio. This supports a **prompt-disjoint** v2 claim only; it does not prove unseen-speaker or unseen-generator generalization. When trustworthy lineage metadata becomes available, pass every relevant field to `--group-columns`. Shared values are treated as OR constraints through connected components, so sharing a speaker **or** source **or** prompt keeps records together.

If trustworthy fields are unavailable, `--allow-id-family-proxy` can test an explicitly acknowledged ID-prefix fallback. For the current metadata that proxy is too coarse to retain every language/label in every split, so the builder refuses it by default. `--allow-missing-strata` exists only for a non-reportable exploratory candidate. Regenerate all train/validation/test feature caches and degraded-audio manifests after adopting v2; never reuse v1 caches. The audit file is SHA-256-bound to the exact split and is verified before training or evaluation.

## Improve the model without contaminating the test set

The repository now compares five validation candidates: the GAT with shared or separate layer normalisation, attention-pooling and mean-pooling baselines, and a feature-robust GAT with noise, time-mask, and channel-dropout augmentation. Compare them on v2 validation only:

```bash
python run_validation_ablations.py \
  --split-file data_pipeline/split_ids_v2.json \
  --cache-dir feature_cache_v2 \
  --checkpoint-root experiments/v2_ablations
```

Select a candidate using validation EER, calibration, and clean/noise/codec robustness. Then run `evaluate_test_split.py` once for the final v2 model. Do not select thresholds, heads, or augmentation settings from the already-observed v1 test results.

For a ready-to-run GPU workflow, use [kaggle/voice_guard_v2_training.ipynb](kaggle/voice_guard_v2_training.ipynb). Its default `smoke` mode checks the complete pipeline on capped data. Change `VOICE_GUARD_TRAIN_MODE` to `full` only for the reportable run, review validation ablations, lock one checkpoint, and then explicitly enable the one-time test cell. See [docs/V2_TRAINING.md](docs/V2_TRAINING.md).

The existing robustness evidence makes noisy/channel audio the highest-priority model problem: the historical model's EER increased sharply under 10 dB MUSAN noise. A future report should include clean, codec, noise, replay/channel, unseen-generator, and real-control results with bootstrap confidence intervals.

## Kaggle and ngrok

Use [docs/KAGGLE_DEPLOYMENT.md](docs/KAGGLE_DEPLOYMENT.md) and the checked-in [Kaggle demo notebook](kaggle/voice_guard_demo.ipynb). Add `NGROK_AUTHTOKEN` as a Kaggle Secret, enable internet and a GPU, then run every cell. The launcher:

- checks out `VOICE_GUARD_REF` and prints the exact resolved commit SHA (`main` by default, or a pinned commit for a frozen demo);
- installs dependencies once;
- starts Streamlit and waits for its health endpoint;
- closes local tunnels before connecting;
- handles endpoint-conflict errors cleanly;
- keeps the hosting cell alive and cleans up in `finally`.

The notebook reads the token from Kaggle Secrets and never prints it. Do not paste a credential into a cell, log, report, or repository file.

## Repository map

- `app.py` - Streamlit interaction and result state
- `codec_robustness.py` - in-memory FFmpeg channel simulations and score comparison
- `voice_guard_core.py` - dependency-light input and decision policies
- `model_pipeline.py` - frontend, heads, training, inference, and evaluation
- `evidence_report.py` - automated analysis PDF
- `data_pipeline/` - frozen splits, leakage audit, and v2 split builder
- `evaluation/` - historical benchmark and robustness artifacts
- `tests/` - safety/policy/report tests

## Important limitations

- A detector result is not proof of identity, authenticity, intent, or deception.
- Noise, compression, replay, unseen generators, and domain shift can cause errors.
- Temperature scaling does not make a score a guaranteed real-world probability; the historical checkpoint's recorded ECE did not improve.
- The v1 test set has already been observed and must not guide further tuning.
- IndicSynth historical checks contain fake samples only, so they measure recall, not false-positive behavior or broad generator generalization.
- The current GAT is quadratic in sequence length; the upload duration cap is a required resource-safety control.

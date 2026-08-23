# Kaggle demo: run and open the UI

The checked-in notebook is self-contained: it clones the repository, resolves the selected Git reference, installs dependencies, starts Streamlit, waits for the health endpoint, and prints a public ngrok URL. The earlier public notebook failed because an ngrok endpoint was already online (`ERR_NGROK_334`), not because Streamlit failed. Its warm-up also submitted one second of digital silence and received a fake verdict; the repaired input policy now rejects that clip before inference.

## One-time Kaggle setup

1. Open `kaggle/voice_guard_demo.ipynb` in Kaggle.
2. In **Add-ons → Secrets**, create `NGROK_AUTHTOKEN`, paste the token as the value, and grant the notebook access.
3. Enable internet access and a GPU accelerator.
4. Run all cells from top to bottom.
5. Open the HTTPS URL printed by the final hosting cell. Keep that cell running while using the UI.

`VOICE_GUARD_REF` defaults to `main`, and the setup cell prints the exact resolved commit SHA for traceability. For a frozen demonstration, change it to an approved 40-character commit SHA before running. The notebook uses `requirements-kaggle.txt`; do not reinstall the same dependencies in later cells.

## Hosting behavior

The checked-in notebook starts Streamlit in one subprocess, polls `/_stcore/health`, creates a new random ngrok URL, and then keeps the cell alive. Stopping or interrupting the cell closes the local tunnel and terminates Streamlit in `finally`.

For a reserved/static ngrok domain, `ngrok.kill()` can close only the local agent. If the same endpoint is active in another Kaggle session or machine, stop that remote endpoint in the ngrok dashboard or use a fresh/random URL. Do not configure pooling unless multiple backends are intentionally serving the same application.

## Verify before sharing the URL

- Upload changes clear the previous verdict and PDF.
- Audio preview matches the selected filename.
- Empty, silent, shorter-than-1-second, longer-than-30-second, and oversized clips are rejected.
- A valid clip shows the three-way result, AI-likelihood model score, permanent disclaimer, and safety action.
- The PDF title says `Automated Analysis Report`, is one readable A4 page for a normal result, and uses the same audio SHA-256 shown in technical details.
- Server errors show a safe public message, while detailed exceptions remain only in the Kaggle logs.

## Useful diagnostics

```bash
curl -fsS http://127.0.0.1:8501/_stcore/health
tail -n 100 /kaggle/working/voice_guard_streamlit.log
```

If model download fails, check Kaggle internet access and Hugging Face availability. If audio decoding fails only for MP3/M4A, confirm the Kaggle image has ffmpeg available for the torchaudio/pydub fallback.

## What this demo does not do

The demo notebook uses the checkpoint committed under `checkpoints/`. It does not train a new model. Run `kaggle/voice_guard_v2_training.ipynb` separately to produce and validate a v2 candidate, then replace the demo checkpoint only after the full validation and one-time test procedure in [V2_TRAINING.md](V2_TRAINING.md).

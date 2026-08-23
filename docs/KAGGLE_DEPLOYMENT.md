# Kaggle deployment checklist

The public v2 notebook failed because an ngrok endpoint was already online (`ERR_NGROK_334`), not because Streamlit failed. Its warm-up also submitted one second of digital silence and received a fake verdict; the repaired input policy now rejects that clip before inference.

## Before launching

1. Verify the local diff and create an approved Git commit only when ready.
2. Put that exact 40-character commit SHA in `VOICE_GUARD_REF` in the notebook. Do not deploy mutable `main`.
3. Enable a Kaggle GPU accelerator. The 300M-parameter frontend is much slower on CPU.
4. Keep the existing ngrok authentication setup if desired, but do not print the credential or store it in repository files.
5. Use `requirements-kaggle.txt`; do not install the same dependencies again in later cells.

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

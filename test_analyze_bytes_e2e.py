"""
End-to-end test of analyze_bytes() on one real clip from the frozen TEST split.

Run in Colab/Kaggle after Step 6 has produced checkpoints/best_model.pt:

    python test_analyze_bytes_e2e.py --checkpoint-dir /path/to/checkpoints

or from a notebook cell:

    import test_analyze_bytes_e2e as t
    t.main(checkpoint_dir="/content/drive/MyDrive/indic-fake-voice-detection/checkpoints")

Streams the HF dataset until it hits a clip whose ID is in the frozen TEST split,
re-encodes it to WAV bytes, and runs analyze_bytes() on those bytes — so the whole
public path (byte decode -> frontend -> head -> calibration -> verdict) is
exercised, not just the internal one.

The test split is used here only for a single-clip smoke test. It is NOT a test-set
evaluation and produces no reportable metric.
"""

import argparse
import io
import json
import sys

import model_pipeline as mp


def encode_wav_bytes(waveform, sample_rate: int) -> bytes:
    """Re-encode a decoded waveform to WAV bytes so analyze_bytes gets real bytes."""
    buf = io.BytesIO()
    try:
        import soundfile as sf

        sf.write(buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except ImportError:
        pass

    import wave

    import numpy as np

    pcm = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def find_one_test_clip(max_scan: int = 4000):
    """Stream until a clip from the frozen TEST split appears. Returns (id, wav_bytes)."""
    _, _, test_ids = mp.load_frozen_split_sets()
    print(f"Frozen test split: {len(test_ids):,} clip IDs")

    ds = mp.open_stream(need_audio=True)
    scanned = 0

    for ex in ds:
        scanned += 1
        if ex["id"] not in test_ids:
            if scanned >= max_scan:
                raise RuntimeError(
                    f"Scanned {scanned:,} examples without hitting a test-split "
                    "clip. Raise --max-scan."
                )
            continue

        clip_id = ex["id"]
        try:
            waveform, sr = mp.decode_audio(ex)
        except Exception as exc:  # noqa: BLE001
            print(f"  {clip_id}: decode failed ({exc}) — trying the next one")
            continue

        print(f"Found test clip after {scanned:,} scanned: {clip_id}")
        print(f"  decoded: {len(waveform):,} samples @ {sr} Hz "
              f"({len(waveform) / sr:.2f}s)")
        return clip_id, encode_wav_bytes(waveform, sr)

    raise RuntimeError("Stream exhausted without finding a test-split clip.")


def main(checkpoint_dir: str = mp.CHECKPOINT_DIR, max_scan: int = 4000) -> int:
    print("=" * 64)
    print("analyze_bytes() END-TO-END TEST — one real test-split clip")
    print("=" * 64)
    print(f"device         : {mp.DEVICE}")
    print(f"checkpoint dir : {checkpoint_dir}")

    # Ground truth for the chosen clip, for context only.
    labels, languages = mp.build_label_lookup()

    clip_id, wav_bytes = find_one_test_clip(max_scan=max_scan)
    print(f"  wav bytes      : {len(wav_bytes):,}")

    true_label = labels.get(clip_id)
    true_language = languages.get(clip_id, "unknown")
    print(f"  ground truth   : label={true_label} "
          f"({'fake' if true_label == mp.LABEL_FAKE else 'real'}), "
          f"language={true_language}")

    print("\n--- first call (loads model from disk) ---")
    import time
    t0 = time.time()
    result = mp.analyze_bytes(wav_bytes, checkpoint_dir=checkpoint_dir)
    t1 = time.time() - t0

    print("\n--- second call (must reuse the cached model) ---")
    t0 = time.time()
    result2 = mp.analyze_bytes(wav_bytes, checkpoint_dir=checkpoint_dir)
    t2 = time.time() - t0
    print(f"  first call: {t1:.2f}s | second call: {t2:.2f}s")

    print("\n" + "=" * 64)
    print("AnalysisResult")
    print("=" * 64)
    print(f"  verdict           : {result.verdict}")
    print(f"  confidence        : {result.confidence:.4f}")
    print(f"  detected_language : {result.detected_language}")
    print(f"  language_eer      : {result.language_eer}")
    print(f"  raw_score         : {result.raw_score:+.4f}")
    print(f"  audio_hash        : {result.audio_hash}")
    print("  metadata:")
    for key, value in result.metadata.items():
        print(f"    {key:<24}: {value}")

    print("\n--- checks ---")
    ok = True

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and condition
        print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    check("verdict is one of real/uncertain/fake",
          result.verdict in ("real", "uncertain", "fake"), result.verdict)
    check("confidence in [0.5, 1.0]", 0.5 <= result.confidence <= 1.0,
          f"{result.confidence:.4f}")
    check("detected_language is not 'unknown' (MMS-LID model detected language)",
          result.detected_language != "unknown", result.detected_language)
    check("language_eer is not None (EER lookup successful)",
          result.language_eer is not None, str(result.language_eer))
    check("audio_hash is 64 hex chars", len(result.audio_hash) == 64)
    check("repeat call is deterministic",
          result.raw_score == result2.raw_score
          and result.audio_hash == result2.audio_hash)
    check("model was cached (2nd call faster)", t2 < t1,
          f"{t1:.2f}s -> {t2:.2f}s")
    check("calibration applied", result.metadata["calibrated"] is True,
          f"temperature={result.metadata['temperature']}")

    verdict_matches = (
        (result.verdict == "fake" and true_label == mp.LABEL_FAKE)
        or (result.verdict == "real" and true_label == mp.LABEL_REAL)
    )
    print(f"\n  ground truth was "
          f"{'fake' if true_label == mp.LABEL_FAKE else 'real'}; "
          f"model said {result.verdict} "
          f"({'agrees' if verdict_matches else 'does not agree'})")
    print("  (one clip proves the plumbing works, not that the model is good)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=mp.CHECKPOINT_DIR)
    parser.add_argument("--max-scan", type=int, default=4000)
    args = parser.parse_args()
    sys.exit(main(args.checkpoint_dir, args.max_scan))

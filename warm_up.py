"""
Model pre-warming script for GPU initialization.

Loads the frozen frontend (wav2vec2-XLS-R-300M) and AASIST head from
checkpoints/best_model.pt, runs one silent dummy inference to fully initialize
everything including JIT compilation, and prints "Model ready" when done.

Run once before the demo on a fresh Kaggle/Colab session to initialize GPU,
then every real analysis will complete in under 10 seconds.

Usage:
    python warm_up.py [--checkpoint-dir /path/to/checkpoints]
"""

import argparse
import sys

import numpy as np
import torch

import model_pipeline as mp


def main(checkpoint_dir: str = mp.CHECKPOINT_DIR) -> int:
    print("=" * 64)
    print("MODEL PRE-WARM: Loading frontend and head, running dummy inference")
    print("=" * 64)
    print(f"device         : {mp.DEVICE}")
    print(f"checkpoint dir : {checkpoint_dir}")

    try:
        # Load the frozen frontend (wav2vec2-XLS-R-300M)
        print("\nLoading frozen frontend...")
        feature_extractor, frontend = mp.load_frontend(device=mp.DEVICE)
        print(f"  Frontend loaded: {mp.FRONTEND_MODEL_NAME}")

        # Load the AASIST head and calibration from checkpoint
        print("Loading inference bundle (head + calibration)...")
        bundle = mp.load_inference_bundle(checkpoint_dir=checkpoint_dir)
        print(f"  Head loaded from epoch {bundle.checkpoint_epoch}")
        print(f"  Temperature: {bundle.temperature:.4f}")

        # Create 1 second of silence at target sample rate
        print(f"\nPreparing dummy inference (1 second of silence @ {mp.TARGET_SAMPLE_RATE} Hz)...")
        silence_duration_s = 1.0
        num_samples = int(silence_duration_s * mp.TARGET_SAMPLE_RATE)
        silent_waveform = np.zeros(num_samples, dtype=np.float32)

        # Run the full inference pipeline once to warm up GPU and JIT
        print("Running dummy inference...")
        with torch.no_grad():
            # Extract features using frontend
            features = mp.extract_features(
                silent_waveform,
                sampling_rate=mp.TARGET_SAMPLE_RATE,
                device=mp.DEVICE,
            )

            # Score using AASIST head
            raw_logit, p_fake = mp.score_waveform(
                silent_waveform,
                sample_rate=mp.TARGET_SAMPLE_RATE,
                bundle=bundle,
            )

        print(f"  Dummy inference complete (logit={raw_logit:.4f}, p_fake={p_fake:.4f})")

        print("\n" + "=" * 64)
        print("Model ready")
        print("=" * 64)
        return 0

    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-warm the model on GPU to initialize everything."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=mp.CHECKPOINT_DIR,
        help=f"Path to checkpoint directory (default: {mp.CHECKPOINT_DIR})",
    )
    args = parser.parse_args()
    sys.exit(main(args.checkpoint_dir))

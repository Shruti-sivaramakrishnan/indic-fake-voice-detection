"""
Model pipeline for AI-generated fake voice detection across 16 Indian languages.

Architecture:
    frozen wav2vec2-XLS-R-300M frontend  ->  AASIST-inspired graph-attention head

Only the head is trained. The frontend is frozen to 0 trainable parameters and is
used purely as a feature extractor.

Provenance
----------
Everything in this file except the PLACEHOLDER sections (11 and 12) is consolidated
from code verified working in Google Colab, including the Step 5 training loop and
checkpointing. Deviations from the verified code are marked [beyond verified] at
their site; hyperparameters not pinned by a verified run are marked [unverified]
in TrainConfig.

Several pieces encode fixes for specific crashes and must not be "simplified" back:

  * GraphAttentionLayer uses separate src/dst score projections that broadcast-add.
    An earlier concat-based implementation materialized a (batch, seq, seq, 2*dim)
    tensor and caused a CUDA OOM. Do not revert to concat-based attention.

  * The torch.nan_to_num call in GraphAttentionLayer.forward is required. Fully
    padded query rows score all -inf, and softmax over an all -inf row is NaN,
    which then propagates through the residual path. Isolated via debug trace.

  * Audio must be decoded with audio_decoder.get_all_samples(). The HF `audio`
    field is a torchcodec AudioDecoder, not a plain array.

  * Passes that do not need audio must call ds.remove_columns("audio") to skip
    decode overhead.

Frozen split
------------
data_pipeline/split_ids.json is Ashitha's frozen stratified split (31,102 clips).
It is loaded as-is and never regenerated. There is deliberately no code path in
this file that can produce a split.

[beyond verified] Feature caches default to float16 storage. The Step 5 smoke test
ran fp32 end to end. fp16 halves both cache size and RAM, which is what makes the
full train split fit at all (see the MEMORY note in Section 6), but it is a
numerical change that no verified run covers. Validate it once on the smoke subset
before trusting a full run, or pass store_dtype=torch.float32 at 2x the cost.

Scope note (15-day compressed timeline): 1 random seed, reduced robustness matrix,
no ONNX quantization this cycle.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


# ============================================================================
# SECTION 1 — Configuration
# ============================================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

SPLIT_FILE = os.path.join(REPO_ROOT, "data_pipeline", "split_ids.json")
# Artifacts live in model_pipeline_artifacts/, NOT model_pipeline/. A directory named
# model_pipeline/ alongside model_pipeline.py is an import trap: once anything imports
# the name while only the directory exists, Python caches a namespace package in
# sys.modules and every later `from model_pipeline import ...` fails with
# "(unknown location)" — even after the .py file appears. Do not recreate that name.
METADATA_FILE = os.path.join(
    REPO_ROOT, "model_pipeline_artifacts", "split_metadata_verified.json"
)
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
FEATURE_CACHE_DIR = os.path.join(REPO_ROOT, "feature_cache")

HF_DATASET = "SherryT997/IndicTTS-Deepfake-Challenge-Data"
FRONTEND_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

TARGET_SAMPLE_RATE = 16000
FRONTEND_DIM = 1024          # XLS-R-300M last_hidden_state width
EXPECTED_TOTAL_CLIPS = 31102

# Label convention, fixed by split_metadata_verified.json: 0 = real (bonafide), 1 = fake.
LABEL_REAL = 0
LABEL_FAKE = 1

# Single seed this cycle (compressed timeline; not the usual 3).
SEED = 42

SPLIT_NAMES = ("train", "validation", "test")


@dataclass
class TrainConfig:
    """
    Hyperparameters for the head. Frontend is frozen and has none.

    Defaults marked [verified] reproduce the Step 5 smoke-test run exactly.
    Defaults marked [unverified] were not pinned by that run — change freely,
    but do not treat them as proven.
    """

    hidden_dim: int = 256            # [verified]
    num_gat_layers: int = 2          # [verified]
    dropout: float = 0.3             # [verified] AASISTHead default used by the run
    learning_rate: float = 1e-4      # [verified]

    # [verified] The run called optim.AdamW(head.parameters(), lr=1e-4) and passed
    # no weight_decay, so it used AdamW's default of 0.01 — NOT 0. Setting this to
    # anything else diverges from the proven configuration.
    weight_decay: float = 0.01

    # [unverified] The smoke test ran 15 epochs on 150 clips. 15 is a smoke-test
    # number, not a tuned one; the full run over 24,881 train clips needs its own
    # value. Left at the verified 15 so nothing is silently assumed.
    num_epochs: int = 15

    # [unverified] The run's DataLoader construction was not captured.
    batch_size: int = 16
    shuffle_train: bool = True

    # [beyond verified] The proven run had NO gradient clipping. 0.0 disables it
    # and reproduces that exactly. Set > 0 only as a deliberate experiment.
    grad_clip: float = 0.0

    seed: int = SEED


# ============================================================================
# SECTION 2 — Determinism and device
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = get_device()


# ============================================================================
# SECTION 3 — Frozen split loading (never regenerated)
# ============================================================================

def load_frozen_split(split_file: str = SPLIT_FILE) -> Dict[str, List[str]]:
    """
    Load Ashitha's frozen split as-is.

    This is the only source of split membership in the pipeline. Nothing here
    recomputes, reshuffles, or re-stratifies it.
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Frozen split not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    if set(splits.keys()) != set(SPLIT_NAMES):
        raise ValueError(
            f"Expected split keys {set(SPLIT_NAMES)}, found {set(splits.keys())}"
        )

    total = sum(len(v) for v in splits.values())
    if total != EXPECTED_TOTAL_CLIPS:
        raise ValueError(
            f"Frozen split has {total:,} clips, expected {EXPECTED_TOTAL_CLIPS:,}. "
            "The split must not be modified."
        )

    return splits


def load_frozen_split_sets(
    split_file: str = SPLIT_FILE,
) -> Tuple[set, set, set]:
    """Frozen split as sets, for O(1) membership tests during the streaming pass."""
    splits = load_frozen_split(split_file)
    return (
        set(splits["train"]),
        set(splits["validation"]),
        set(splits["test"]),
    )


def load_verified_metadata(
    metadata_file: str = METADATA_FILE,
) -> Dict[str, List[Tuple[str, str, int]]]:
    """
    Load Step 2 output: {split: [[clip_id, language, label], ...]}.

    Note this file is grouped by language while split_ids.json is shuffled, so the
    two are NOT index-aligned. Always join on clip_id, never on position.
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Verified metadata not found: {metadata_file}")

    with open(metadata_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        split: [(row[0], row[1], int(row[2])) for row in rows]
        for split, rows in raw.items()
    }


def build_label_lookup(
    metadata: Optional[Dict[str, List[Tuple[str, str, int]]]] = None,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Return (clip_id -> label, clip_id -> language) across all splits."""
    if metadata is None:
        metadata = load_verified_metadata()

    labels: Dict[str, int] = {}
    languages: Dict[str, str] = {}
    for rows in metadata.values():
        for clip_id, language, label in rows:
            labels[clip_id] = label
            languages[clip_id] = language

    return labels, languages


def assert_split_integrity(
    splits: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    Cheap in-process restatement of data_pipeline/leakage_audit.py, so a training
    run cannot silently proceed on a corrupted split. Does not replace the audit.
    """
    if splits is None:
        splits = load_frozen_split()

    train, val, test = (set(splits[name]) for name in SPLIT_NAMES)

    for name in SPLIT_NAMES:
        if len(splits[name]) != len(set(splits[name])):
            raise ValueError(f"Duplicate clip IDs within split '{name}'.")

    if train & val or train & test or val & test:
        raise ValueError("Split leakage: clip IDs cross split boundaries.")

    if len(train | val | test) != EXPECTED_TOTAL_CLIPS:
        raise ValueError("Unique clip ID count does not match the frozen split.")


# ============================================================================
# SECTION 4 — ID -> audio resolution (HF streaming; slow but proven)
# ============================================================================
# Streaming .shuffle() only shuffles within a small buffer and offers no random
# access, so there is no way to look a clip up by ID. Every pass scans the full
# ~31k-clip stream (~20+ min). Cache aggressively downstream rather than
# repeating passes.

def open_stream(need_audio: bool = True):
    """Open the HF dataset in streaming mode, dropping audio when not needed."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    if not need_audio:
        # Skips torchcodec decode entirely — use for metadata-only passes.
        ds = ds.remove_columns("audio")

    return ds


def stream_and_filter_by_ids(
    train_ids: set,
    val_ids: set,
    test_ids: set,
    need_audio: bool = True,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Full pass over the HF stream, bucketing examples into train/val/test by
    matching against the frozen ID sets.

    Set need_audio=False for metadata-only passes (much faster, no decode).
    """
    ds = open_stream(need_audio=need_audio)

    train_examples: List[dict] = []
    val_examples: List[dict] = []
    test_examples: List[dict] = []

    for ex in ds:
        cid = ex["id"]
        if cid in train_ids:
            train_examples.append(ex)
        elif cid in val_ids:
            val_examples.append(ex)
        elif cid in test_ids:
            test_examples.append(ex)

    return train_examples, val_examples, test_examples


def decode_audio(example: dict) -> Tuple[np.ndarray, int]:
    """
    The `audio` field is a torchcodec AudioDecoder, not a plain array — it must be
    read via .get_all_samples(). Returns (waveform, sample_rate).
    """
    audio_decoder = example["audio"]
    decoded = audio_decoder.get_all_samples()
    waveform = decoded.data.numpy().squeeze()
    sr = decoded.sample_rate
    return waveform, sr


def ensure_target_sample_rate(
    waveform: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Guard against a silent correctness bug: XLS-R expects 16 kHz, and feeding it
    another rate degrades features without raising anything.
    """
    if sample_rate == target_sample_rate:
        return waveform

    import torchaudio

    resampled = torchaudio.functional.resample(
        torch.from_numpy(waveform).float(),
        orig_freq=sample_rate,
        new_freq=target_sample_rate,
    )
    return resampled.numpy()


# ============================================================================
# SECTION 5 — Frozen frontend
# ============================================================================
# Loaded lazily. Importing this module must not download 1.2 GB of weights or
# claim GPU memory as a side effect — that only happens on first actual use.

_feature_extractor = None
_frontend = None


def load_frontend(
    model_name: str = FRONTEND_MODEL_NAME,
    device: Optional[str] = None,
) -> Tuple[Any, nn.Module]:
    """
    Load and freeze the wav2vec2-XLS-R-300M frontend. Cached after first call.

    Returns (feature_extractor, frozen_frontend). Verified to freeze to
    0 trainable parameters.
    """
    global _feature_extractor, _frontend

    if _feature_extractor is not None and _frontend is not None:
        return _feature_extractor, _frontend

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    device = device or DEVICE

    _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    frontend = Wav2Vec2Model.from_pretrained(model_name)

    for param in frontend.parameters():
        param.requires_grad = False
    frontend.eval()
    _frontend = frontend.to(device)

    return _feature_extractor, _frontend


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def verify_frontend_frozen() -> int:
    """Assert the frontend has 0 trainable params. Returns the count."""
    _, frontend = load_frontend()
    n = count_trainable_params(frontend)
    if n != 0:
        raise RuntimeError(f"Frontend is not frozen: {n:,} trainable parameters.")
    return n


def extract_features(
    audio_array: np.ndarray,
    sampling_rate: int = TARGET_SAMPLE_RATE,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Returns (seq_len, 1024) frozen frontend hidden states for one clip, on CPU."""
    device = device or DEVICE
    feature_extractor, frontend = load_frontend(device=device)

    audio_array = ensure_target_sample_rate(audio_array, sampling_rate)

    inputs = feature_extractor(
        audio_array,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = frontend(input_values)

    hidden_states = outputs.last_hidden_state.squeeze(0)
    return hidden_states.cpu()


def frontend_uses_attention_mask() -> bool:
    """
    Whether this frontend's feature extractor returns an attention mask.

    Decisive for batching. XLS-R (a wav2vec2-large variant) returns one; the
    wav2vec2-base checkpoints do not. Without a mask, a padded batch is NOT
    numerically equivalent to single-item extraction — padding leaks into both
    the per-utterance normalization and the self-attention — so batching is
    unsafe and must be refused rather than silently approximated.
    """
    feature_extractor, _ = load_frontend()
    return bool(getattr(feature_extractor, "return_attention_mask", False))


def extract_features_batch(
    audio_arrays: Sequence[np.ndarray],
    sampling_rates: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Batched frozen-frontend extraction. Returns one (seq_len_i, 1024) CPU tensor
    per input, each trimmed to its own true length.

    Numerically equivalent to calling extract_features() per clip, not an
    approximation of it, because:

      * the feature extractor pads and returns an attention_mask, and applies its
        zero-mean/unit-variance normalization per sequence over the unpadded
        length (not over the padded tensor);
      * the mask is passed to the model, so padded frames are excluded from
        self-attention;
      * each output is sliced back to its own frame count via the model's
        conv-stack length formula, so no padded frame is ever stored.

    Falls back to per-clip extraction if the feature extractor has no attention
    mask — correctness first.
    """
    device = device or DEVICE
    feature_extractor, frontend = load_frontend(device=device)

    if sampling_rates is None:
        sampling_rates = [TARGET_SAMPLE_RATE] * len(audio_arrays)

    arrays = [
        ensure_target_sample_rate(a, sr)
        for a, sr in zip(audio_arrays, sampling_rates)
    ]

    if len(arrays) == 1:
        return [extract_features(arrays[0], TARGET_SAMPLE_RATE, device)]

    if not frontend_uses_attention_mask():
        # Padding would change the result; extract one at a time instead.
        return [extract_features(a, TARGET_SAMPLE_RATE, device) for a in arrays]

    inputs = feature_extractor(
        arrays,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return [extract_features(a, TARGET_SAMPLE_RATE, device) for a in arrays]
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = frontend(input_values, attention_mask=attention_mask)

    hidden = outputs.last_hidden_state  # (batch, padded_frames, 1024)

    # True per-item frame counts after the conv stack's downsampling.
    out_lens = frontend._get_feat_extract_output_lengths(attention_mask.sum(-1))
    out_lens = [int(x) for x in out_lens]

    return [hidden[i, : out_lens[i]].cpu() for i in range(len(arrays))]


def _plan_batches(
    lengths: Sequence[int],
    max_items: int,
    max_padded_samples: int,
) -> List[List[int]]:
    """
    Group indices into length-sorted batches bounded by both item count and
    padded-sample budget.

    Sorting by length keeps padding waste low. The sample budget stops one long
    clip from turning a full batch into a multi-GB attention tensor — attention is
    quadratic in sequence length, so a 30s clip batched with 15 others is exactly
    how this OOMs.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])

    batches: List[List[int]] = []
    current: List[int] = []
    current_max = 0

    for idx in order:
        candidate_max = max(current_max, lengths[idx])
        if current and (
            len(current) + 1 > max_items
            or (len(current) + 1) * candidate_max > max_padded_samples
        ):
            batches.append(current)
            current, current_max = [idx], lengths[idx]
        else:
            current.append(idx)
            current_max = candidate_max

    if current:
        batches.append(current)

    return batches


# ============================================================================
# SECTION 6 — Feature cache
# ============================================================================
# A full streaming pass plus frontend extraction is the expensive step. Cache the
# extracted features to disk so training/eval reruns skip both.
#
# MEMORY — read before running this at full scale.
#
# There are two separate memory problems here, and only one of them is about audio.
#
#   1. Raw audio. stream_and_filter_by_ids(need_audio=True) collects every matching
#      example, decoder object included, into a list before extraction starts.
#      build_feature_caches_streaming() below fixes this: one clip's audio is
#      resident at a time and is dropped as soon as its features exist.
#
#   2. The features themselves — the larger problem. XLS-R emits 50 frames/sec at
#      1024 dims, so fp32 features cost ~0.20 MB per second of audio. Across 24,881
#      train clips that is roughly 15 GB at a 3s average and 25 GB at 5s. Colab's
#      free tier has ~12.7 GB. Accumulating features in one Python list therefore
#      OOMs regardless of how carefully the audio is handled, and periodically
#      re-saving that growing list neither frees memory nor keeps I/O bounded
#      (re-serializing the whole list every N clips is quadratic).
#
# The fix for (2) is sharding: fill a buffer of shard_size clips, write it, clear
# it. Peak RAM is one shard, and each shard is written exactly once. Training then
# reads shards back through ShardedFeatureDataset rather than one giant list.

def build_feature_cache(
    examples: Sequence[dict],
    label_lookup: Dict[str, int],
    device: Optional[str] = None,
    progress_every: int = 250,
) -> List[Tuple[torch.Tensor, int]]:
    """Extract frozen features for streamed examples -> [(feat_tensor, label), ...]."""
    cached: List[Tuple[torch.Tensor, int]] = []

    for i, ex in enumerate(examples):
        clip_id = ex["id"]
        if clip_id not in label_lookup:
            raise KeyError(f"Clip {clip_id} is not in the verified metadata.")

        waveform, sr = decode_audio(ex)
        feats = extract_features(waveform, sampling_rate=sr, device=device)
        cached.append((feats, label_lookup[clip_id]))

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  extracted {i + 1}/{len(examples)}")

    return cached


def save_feature_cache(
    cached_features: List[Tuple[torch.Tensor, int]],
    split_name: str,
    cache_dir: str = FEATURE_CACHE_DIR,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{split_name}_features.pt")
    torch.save(cached_features, path)
    return path


def load_feature_cache(
    split_name: str,
    cache_dir: str = FEATURE_CACHE_DIR,
) -> List[Tuple[torch.Tensor, int]]:
    path = os.path.join(cache_dir, f"{split_name}_features.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No feature cache for '{split_name}': {path}")
    return torch.load(path)


# ---------------------------------------------------------------------------
# Sharded streaming cache
# ---------------------------------------------------------------------------

def shard_path(split_name: str, shard_idx: int, cache_dir: str = FEATURE_CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"{split_name}_shard_{shard_idx:05d}.pt")


def manifest_path(split_name: str, cache_dir: str = FEATURE_CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"{split_name}_manifest.json")


def failed_clips_path(cache_dir: str = FEATURE_CACHE_DIR) -> str:
    return os.path.join(cache_dir, "failed_clips.json")


# How many individual failures to print before falling back to the log file only.
MAX_PRINTED_FAILURES = 20

# If this fraction of attempted clips fails, the problem is systemic (API change,
# dead network, wrong column) rather than a few bad files — stop instead of
# burning hours producing an unusable cache.
ABORT_FAILURE_RATE = 0.25
ABORT_FAILURE_MIN_ATTEMPTS = 50


def read_failed_clips(cache_dir: str = FEATURE_CACHE_DIR) -> List[Dict[str, str]]:
    path = failed_clips_path(cache_dir)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("failures", [])


def write_failed_clips(
    new_failures: List[Dict[str, str]],
    cache_dir: str = FEATURE_CACHE_DIR,
) -> str:
    """
    Append this run's failures to the cumulative log.

    History is kept rather than overwritten: a clip that fails on every attempt is
    a genuinely bad file, while one that fails once and later succeeds was a
    transient network read. That distinction is only visible across runs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = failed_clips_path(cache_dir)

    existing = read_failed_clips(cache_dir)
    combined = existing + new_failures

    by_clip: Dict[str, int] = {}
    for record in combined:
        by_clip[record["clip_id"]] = by_clip.get(record["clip_id"], 0) + 1

    payload = {
        "total_failure_events": len(combined),
        "distinct_clips": len(by_clip),
        "repeat_offenders": sorted(
            (cid for cid, n in by_clip.items() if n > 1),
        ),
        "failures": combined,
    }

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def read_manifest(
    split_name: str, cache_dir: str = FEATURE_CACHE_DIR
) -> Optional[Dict[str, Any]]:
    path = manifest_path(split_name, cache_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(
    split_name: str,
    shards: List[Dict[str, Any]],
    store_dtype: str,
    complete: bool,
    cache_dir: str = FEATURE_CACHE_DIR,
    capped: Optional[int] = None,
    expected_clips: Optional[int] = None,
    failed_clip_ids: Optional[Sequence[str]] = None,
) -> str:
    """
    Rewritten after every shard, so a crash leaves a manifest describing exactly
    the shards that finished writing.

    `capped` records a max_clips_per_split rehearsal limit. Such a cache is
    "complete" for its own purposes but is NOT the full split — callers that
    intend to train for real must check this.

    `expected_clips` and `failed_clip_ids` make a shortfall auditable: a complete
    cache satisfies num_clips + num_failed == expected_clips, so a split that
    quietly under-delivers cannot look identical to one that did not.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = manifest_path(split_name, cache_dir)

    num_clips = sum(s["count"] for s in shards)
    failed_clip_ids = list(failed_clip_ids or [])

    payload = {
        "split": split_name,
        "store_dtype": store_dtype,
        "complete": complete,
        "capped": capped,
        "num_clips": num_clips,
        "expected_clips": expected_clips,
        "num_failed": len(failed_clip_ids),
        "failed_clip_ids": failed_clip_ids,
        "accounted_for": (
            None if expected_clips is None
            else num_clips + len(failed_clip_ids) == expected_clips
        ),
        "shards": shards,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)  # atomic: never leave a half-written manifest
    return path


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def build_feature_caches_streaming(
    targets: Dict[str, set],
    label_lookup: Dict[str, int],
    language_lookup: Optional[Dict[str, str]] = None,
    cache_dir: str = FEATURE_CACHE_DIR,
    shard_size: int = 500,
    store_dtype: torch.dtype = torch.float16,
    device: Optional[str] = None,
    resume: bool = True,
    progress_every: int = 100,
    max_clips_per_split: Optional[int] = None,
    extract_batch_size: int = 8,
    batch_sort_factor: int = 4,
    # 120 s of padded audio per forward pass. Self-attention is quadratic in
    # sequence length, so this is what stops 8 x 30 s clips (T=1500 padded frames,
    # ~1 GB of transient attention per layer) from landing in one batch.
    max_batch_samples: int = 120 * TARGET_SAMPLE_RATE,
) -> Dict[str, Dict[str, Any]]:
    """
    Single-pass, memory-bounded feature cache builder.

    Opens the HF stream ONCE and builds caches for every split in `targets`
    simultaneously (e.g. {"train": train_ids, "validation": val_ids}), because a
    full pass costs ~20+ minutes and there is no random access to amortise it.

    Per matching example:
        decode audio -> hold in a small pending buffer -> extract in batches
        -> cast to store_dtype -> append tuple -> drop the waveforms.

    Peak RAM is one shard of features plus at most
    extract_batch_size * batch_sort_factor raw waveforms (tens of MB), regardless
    of how many clips the split holds.

    Batching matters: the frozen frontend is a 300M-parameter model, and a
    single-clip forward pass leaves the GPU almost entirely idle. Because it is
    frozen and in eval mode, batching computes exactly the same features — see
    extract_features_batch for why padding does not contaminate the result.

    Args:
        targets: {split_name: set_of_clip_ids}. Splits are disjoint (guaranteed by
            the frozen split), so each streamed example matches at most one.
        store_dtype: float16 halves cache size and RAM. See the [beyond verified]
            note in the module docs — the smoke test ran fp32.
        resume: skip clip IDs already present in a previous run's shards.
        max_clips_per_split: cap per split, for a cheap end-to-end rehearsal.
        extract_batch_size: clips per frontend forward pass.
        batch_sort_factor: how many batches to buffer before length-sorting them.
            Higher means less padding waste and more raw audio held at once.
        max_batch_samples: padded-sample budget per forward pass. Attention is
            quadratic in length, so this caps the cost of a long outlier clip.

    Returns:
        {split_name: manifest_dict}
    """
    os.makedirs(cache_dir, exist_ok=True)
    device = device or DEVICE
    dtype_name = str(store_dtype).replace("torch.", "")

    # --- resume bookkeeping ------------------------------------------------
    shards: Dict[str, List[Dict[str, Any]]] = {name: [] for name in targets}
    done_ids: Dict[str, set] = {name: set() for name in targets}

    for name in targets:
        existing = read_manifest(name, cache_dir) if resume else None
        if not existing:
            continue
        if existing.get("store_dtype") != dtype_name:
            print(
                f"[{name}] existing cache is {existing.get('store_dtype')} but "
                f"store_dtype={dtype_name} was requested — ignoring it and "
                f"rebuilding from scratch."
            )
            continue
        for shard in existing["shards"]:
            if os.path.exists(os.path.join(cache_dir, shard["file"])):
                shards[name].append(shard)
                done_ids[name].update(shard["clip_ids"])
        if done_ids[name]:
            print(f"[{name}] resuming — {len(done_ids[name]):,} clips already cached.")

    # Remaining work per split. Note the stream still has to be scanned in full;
    # resume saves the GPU extraction, not the scan.
    remaining = {
        name: ids - done_ids[name]
        for name, ids in targets.items()
    }
    total_wanted = {
        name: (
            len(ids) if max_clips_per_split is None
            else min(len(ids), max_clips_per_split)
        )
        for name, ids in targets.items()
    }

    if all(not r for r in remaining.values()):
        print("All requested splits are already cached — nothing to do.")
        return {
            name: read_manifest(name, cache_dir) or {}
            for name in targets
        }

    buffers: Dict[str, List[Tuple[torch.Tensor, int]]] = {n: [] for n in targets}
    buffer_ids: Dict[str, List[str]] = {n: [] for n in targets}

    # counts = clips actually cached. queued = clips accepted for extraction
    # (decoded OK, awaiting the batched forward). They diverge when extraction
    # fails, so the cap and the progress display must not share one counter.
    counts: Dict[str, int] = {n: len(done_ids[n]) for n in targets}
    queued: Dict[str, int] = {n: len(done_ids[n]) for n in targets}
    bytes_written: Dict[str, int] = {n: 0 for n in targets}

    # Per-clip failures. Recorded rather than raised: across ~28k network-fetched
    # clips a non-zero decode failure rate is expected, and one unreadable file
    # must not end the pass.
    failures: Dict[str, List[Dict[str, str]]] = {n: [] for n in targets}
    failure_printed = 0

    def record_failure(
        split_name: str, clip_id: str, stage: str, exc: BaseException
    ) -> None:
        nonlocal failure_printed

        record = {
            "clip_id": clip_id,
            "split": split_name,
            "language": (language_lookup or {}).get(clip_id, "unknown"),
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "seen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        failures[split_name].append(record)

        if failure_printed < MAX_PRINTED_FAILURES:
            failure_printed += 1
            print(
                f"  SKIP [{split_name}] {clip_id} ({record['language']}) "
                f"{stage} failed: {record['error_type']}: {record['error'][:160]}"
            )
            if failure_printed == MAX_PRINTED_FAILURES:
                print(
                    f"  (further failures suppressed; full list in "
                    f"{os.path.basename(failed_clips_path(cache_dir))})"
                )

    def total_failures() -> int:
        return sum(len(v) for v in failures.values())

    def flush(name: str, final: bool = False, limit: Optional[int] = None) -> None:
        """
        Write one shard and free it. The only place features leave RAM.

        `limit` writes exactly that many clips and keeps the remainder buffered —
        batched extraction can overshoot shard_size, and shards must stay evenly
        sized for the sampler's shuffle window to mean what it says.
        """
        if not buffers[name]:
            if final:
                write_manifest(
                    name, shards[name], dtype_name, True, cache_dir,
                    capped=max_clips_per_split,
                    expected_clips=total_wanted[name],
                    failed_clip_ids=[r["clip_id"] for r in failures[name]],
                )
            return

        take = len(buffers[name]) if limit is None else min(limit, len(buffers[name]))
        payload = buffers[name][:take]
        payload_ids = buffer_ids[name][:take]

        idx = len(shards[name])
        path = shard_path(name, idx, cache_dir)
        torch.save(payload, path)

        size = os.path.getsize(path)
        bytes_written[name] += size
        shards[name].append({
            "file": os.path.basename(path),
            "count": len(payload),
            "bytes": size,
            "clip_ids": list(payload_ids),
        })

        # Manifest is rewritten here so a crash mid-run leaves a consistent,
        # resumable record of every shard that made it to disk.
        write_manifest(
            name, shards[name], dtype_name, final, cache_dir,
            capped=max_clips_per_split,
            expected_clips=total_wanted[name],
            failed_clip_ids=[r["clip_id"] for r in failures[name]],
        )

        del buffers[name][:take]
        del buffer_ids[name][:take]

    # Timing breakdown. The whole point is to separate the FIXED cost of scanning
    # the stream (network + parquet decode, paid once regardless of how many clips
    # match) from the PER-CLIP cost of decode + GPU extraction. Only the latter
    # scales with target count, so the two must not be averaged together.
    timings = {"stream": 0.0, "decode": 0.0, "extract": 0.0, "save": 0.0}

    # Raw audio waiting to be extracted as a batch.
    pending: List[Tuple[str, str, np.ndarray]] = []

    def run_pending() -> None:
        """Extract all pending clips in length-sorted, budget-bounded batches."""
        if not pending:
            return

        t0 = time.time()
        lengths = [len(w) for _, _, w in pending]

        for group in _plan_batches(lengths, extract_batch_size, max_batch_samples):
            try:
                feats_list: List[Optional[torch.Tensor]] = list(
                    extract_features_batch(
                        [pending[i][2] for i in group],
                        [TARGET_SAMPLE_RATE] * len(group),
                        device=device,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — isolate, don't lose the batch
                # A batch-level failure (CUDA OOM, one pathological tensor) would
                # otherwise discard every clip in the group. Retry individually so
                # only the genuinely bad clip is lost.
                print(
                    f"  batch of {len(group)} failed "
                    f"({type(exc).__name__}: {str(exc)[:120]}) — "
                    "retrying clips individually"
                )
                feats_list = []
                for i in group:
                    split_name, clip_id, waveform = pending[i]
                    try:
                        feats_list.append(
                            extract_features(waveform, TARGET_SAMPLE_RATE, device)
                        )
                    except Exception as inner:  # noqa: BLE001
                        record_failure(split_name, clip_id, "extract", inner)
                        feats_list.append(None)

            for i, feats in zip(group, feats_list):
                if feats is None:
                    continue
                split_name, clip_id, _ = pending[i]
                buffers[split_name].append(
                    (feats.to(store_dtype), label_lookup[clip_id])
                )
                buffer_ids[split_name].append(clip_id)
                counts[split_name] += 1

        timings["extract"] += time.time() - t0

        pending.clear()

        for name in targets:
            while len(buffers[name]) >= shard_size:
                t1 = time.time()
                flush(name, limit=shard_size)
                timings["save"] += time.time() - t1

    def unresolved(name: str) -> set:
        """
        Target IDs neither cached nor attempted-and-failed.

        Failed clips stay in `remaining` so a later resume retries them (a
        truncated read is usually transient), but they must not hold the split
        open as "incomplete" forever — completeness means every clip was
        accounted for, not that every clip succeeded.
        """
        return remaining[name] - {r["clip_id"] for r in failures[name]}

    started = time.time()
    scanned = 0
    matched = 0
    attempted = 0

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print(
            "  WARNING: no GPU detected. The frozen frontend forward pass is "
            "roughly 20x slower on CPU and will dominate the run. Check the Colab "
            "runtime type before committing to a full extraction."
        )
    print(
        f"Batching: up to {extract_batch_size} clips/forward "
        f"(sorted by length, {max_batch_samples:,} padded-sample budget)"
    )
    print(
        "Streaming pass: "
        + ", ".join(f"{n} {len(remaining[n]):,} to extract" for n in targets)
    )

    t_stream = time.time()
    ds = open_stream(need_audio=True)
    iterator = iter(ds)
    timings["stream"] += time.time() - t_stream

    try:
        while True:
            # Timed separately: this is where network I/O and parquet decoding land.
            t_stream = time.time()
            try:
                ex = next(iterator)
            except StopIteration:
                break
            finally:
                timings["stream"] += time.time() - t_stream

            scanned += 1

            clip_id = ex["id"]
            split_name = None
            for name in targets:
                if clip_id in remaining[name]:
                    split_name = name
                    break

            if split_name is None:
                continue

            if (
                max_clips_per_split is not None
                and queued[split_name] >= max_clips_per_split
            ):
                remaining[split_name].discard(clip_id)
                continue

            if clip_id not in label_lookup:
                raise KeyError(f"Clip {clip_id} is not in the verified metadata.")

            # Decode is the failure-prone step: the bytes arrived over the network
            # and torchcodec raises on a truncated or corrupt buffer. Catching here
            # — before the clip is queued — is also what keeps batching safe: a
            # clip that cannot be decoded never enters a batch, so the batch
            # correctness guarantees are untouched.
            t_decode = time.time()
            try:
                waveform, sr = decode_audio(ex)
                waveform = ensure_target_sample_rate(waveform, sr)
            except Exception as exc:  # noqa: BLE001 — one bad file must not end the pass
                timings["decode"] += time.time() - t_decode
                attempted += 1
                # Not discarded from `remaining`: a transient read error should be
                # retried on the next resume. It is dropped from this pass only.
                record_failure(split_name, clip_id, "decode", exc)

                if (
                    attempted >= ABORT_FAILURE_MIN_ATTEMPTS
                    and total_failures() / attempted >= ABORT_FAILURE_RATE
                ):
                    raise RuntimeError(
                        f"{total_failures():,} of {attempted:,} clips failed "
                        f"({total_failures() / attempted * 100:.0f}%). That is a "
                        "systemic failure, not a few bad files — stopping. "
                        f"See {failed_clips_path(cache_dir)}."
                    ) from exc
                continue
            timings["decode"] += time.time() - t_decode

            pending.append((split_name, clip_id, waveform))
            del waveform

            remaining[split_name].discard(clip_id)
            queued[split_name] += 1
            attempted += 1
            matched += 1

            if len(pending) >= extract_batch_size * batch_sort_factor:
                run_pending()

            if progress_every and matched % progress_every == 0:
                elapsed = time.time() - started
                rate = matched / elapsed if elapsed else 0.0
                left = sum(max(total_wanted[n] - counts[n], 0) for n in targets)
                eta = left / rate if rate else 0.0
                done_str = ", ".join(
                    f"{n} {counts[n]:,}/{total_wanted[n]:,}" for n in targets
                )
                cached_mb = sum(bytes_written.values()) / 1e6
                print(
                    f"  {done_str} | scanned {scanned:,} | "
                    f"{rate:.1f} clips/s | elapsed {_format_duration(elapsed)} | "
                    f"eta {_format_duration(eta)} | cache {cached_mb:,.0f} MB"
                )

            if all(not unresolved(n) for n in targets):
                print("  all target IDs accounted for — ending stream early.")
                break

    finally:
        # Runs on success, on KeyboardInterrupt, and on exception, so neither a
        # pending batch nor a partial buffer nor the failure log is silently lost.
        try:
            run_pending()
        finally:
            for name in targets:
                while len(buffers[name]) > shard_size:
                    flush(name, limit=shard_size)
                flush(name, final=not unresolved(name))

            new_failures = [r for name in targets for r in failures[name]]
            if new_failures:
                write_failed_clips(new_failures, cache_dir)

    elapsed = time.time() - started
    print(
        f"\nStreaming pass complete in {_format_duration(elapsed)} "
        f"— scanned {scanned:,}, extracted {matched:,}."
    )
    for name in targets:
        never_seen = len(unresolved(name))
        n_failed = len(failures[name])
        status = "COMPLETE" if not never_seen else f"INCOMPLETE ({never_seen:,} unseen)"
        print(
            f"  [{name}] {counts[name]:,} cached, {n_failed:,} failed, "
            f"{len(shards[name])} shards, "
            f"{bytes_written[name] / 1e9:.2f} GB — {status}"
        )
        if never_seen:
            print(
                f"    WARNING: {never_seen:,} {name} IDs never appeared in the "
                "stream. Re-run with resume=True to retry them."
            )

    n_failed_total = total_failures()
    if n_failed_total:
        print("\n--- DECODE/EXTRACT FAILURES ---")
        print(
            f"  {n_failed_total:,} of {attempted:,} attempted clips failed "
            f"({n_failed_total / max(attempted, 1) * 100:.2f}%)"
        )
        for name in targets:
            if not failures[name]:
                continue
            expected = total_wanted[name]
            print(
                f"  [{name}] {len(failures[name]):,} failed — "
                f"{counts[name]:,} cached + {len(failures[name]):,} failed "
                f"= {counts[name] + len(failures[name]):,} of {expected:,} expected"
            )
            by_stage: Dict[str, int] = {}
            by_error: Dict[str, int] = {}
            by_language: Dict[str, int] = {}
            for record in failures[name]:
                by_stage[record["stage"]] = by_stage.get(record["stage"], 0) + 1
                by_error[record["error_type"]] = (
                    by_error.get(record["error_type"], 0) + 1
                )
                by_language[record["language"]] = (
                    by_language.get(record["language"], 0) + 1
                )
            print(f"      by stage    : {by_stage}")
            print(f"      by error    : {by_error}")
            if len(by_language) <= 8:
                print(f"      by language : {by_language}")

        print(f"  full log: {failed_clips_path(cache_dir)}")

        repeats = [
            r["clip_id"]
            for r in read_failed_clips(cache_dir)
        ]
        seen_counts: Dict[str, int] = {}
        for cid in repeats:
            seen_counts[cid] = seen_counts.get(cid, 0) + 1
        persistent = sorted(cid for cid, n in seen_counts.items() if n > 1)
        if persistent:
            print(
                f"  {len(persistent):,} clip(s) have now failed more than once — "
                "likely genuinely corrupt rather than transient: "
                f"{persistent[:5]}{'...' if len(persistent) > 5 else ''}"
            )
        else:
            print(
                "  all failures are first-time — re-running with resume=True will "
                "retry them, and transient read errors usually succeed."
            )
    else:
        print("\nNo decode or extraction failures.")

    _report_timings(timings, elapsed, scanned, matched, targets)

    return {name: read_manifest(name, cache_dir) or {} for name in targets}


def _report_timings(
    timings: Dict[str, float],
    elapsed: float,
    scanned: int,
    matched: int,
    targets: Dict[str, set],
) -> None:
    """
    Print where the time actually went, and project the full run from it.

    The projection is the reason this exists. Dividing total elapsed time by
    matched clips conflates two very different costs and badly overestimates the
    full run whenever the scan dominates: scanning the stream is paid once no
    matter how many clips you keep, while decode+extract is paid per kept clip.
    """
    accounted = sum(timings.values())
    other = max(elapsed - accounted, 0.0)

    print("\n--- TIMING BREAKDOWN ---")
    rows = [
        ("stream i/o (network + parquet)", timings["stream"]),
        ("audio decode", timings["decode"]),
        ("frontend extraction (GPU)", timings["extract"]),
        ("shard writes", timings["save"]),
        ("other (matching, bookkeeping)", other),
    ]
    for label, value in rows:
        share = value / elapsed * 100 if elapsed else 0.0
        print(f"  {label:<32} {_format_duration(value):>9}  {share:5.1f}%")

    if not scanned or not matched:
        return

    per_scanned = timings["stream"] / scanned
    per_matched = (timings["decode"] + timings["extract"]) / matched

    print("\n--- UNIT COSTS ---")
    print(f"  fixed:    {per_scanned * 1000:7.1f} ms per streamed example (scan)")
    print(f"  variable: {per_matched * 1000:7.1f} ms per extracted clip (decode+GPU)")

    full_targets = sum(len(v) for v in targets.values())
    projected = EXPECTED_TOTAL_CLIPS * per_scanned + full_targets * per_matched

    print("\n--- FULL-RUN PROJECTION ---")
    print(
        f"  scan {EXPECTED_TOTAL_CLIPS:,} examples : "
        f"{_format_duration(EXPECTED_TOTAL_CLIPS * per_scanned)}"
    )
    print(
        f"  extract {full_targets:,} clips    : "
        f"{_format_duration(full_targets * per_matched)}"
    )
    print(f"  projected total          : {_format_duration(projected)}")

    naive = elapsed / matched * full_targets
    print(
        f"  (naive elapsed/matched extrapolation would say "
        f"{_format_duration(naive)} — it double-counts the fixed scan)"
    )


# ---------------------------------------------------------------------------
# Cache audit and repair
# ---------------------------------------------------------------------------
# A cache that has been copied between machines, partially re-downloaded, and
# resumed across sessions can drift from its manifest in several independent
# ways. These tools find and fix that drift without re-extracting anything.

def audit_feature_cache(
    split_name: str,
    cache_dir: str = FEATURE_CACHE_DIR,
    target_ids: Optional[set] = None,
    compare_tensors: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Inspect a cache for every way it can disagree with its manifest.

    Checks, independently:
      * clip IDs appearing more than once (within or across shards)
      * whether duplicate copies hold identical tensors (benign double-copy) or
        different ones (something worse)
      * clips listed as failed that are actually present in a shard — a clip that
        failed once and succeeded on a later resume leaves a stale failure entry
      * cached clips that are not in the split's target ID set at all
      * target clips that are neither cached nor recorded as failed
      * manifest shard counts that disagree with the shard payloads
      * manifest shards missing from disk, and shard files not in the manifest
    """
    manifest = read_manifest(split_name, cache_dir)
    if manifest is None:
        raise FileNotFoundError(f"No manifest for '{split_name}' in {cache_dir}.")

    if target_ids is None:
        splits = load_frozen_split()
        target_ids = set(splits[split_name])

    occurrences: Dict[str, List[Tuple[int, int]]] = {}
    count_mismatches: List[Tuple[int, int, int]] = []
    files_missing: List[str] = []
    shard_payload_cache: Dict[int, Any] = {}

    def get_shard(idx: int):
        if idx not in shard_payload_cache:
            path = os.path.join(cache_dir, manifest["shards"][idx]["file"])
            shard_payload_cache[idx] = torch.load(path)
        return shard_payload_cache[idx]

    for shard_idx, shard in enumerate(manifest["shards"]):
        path = os.path.join(cache_dir, shard["file"])
        if not os.path.exists(path):
            files_missing.append(shard["file"])
            continue

        clip_ids = shard["clip_ids"]

        # Trust but verify: the payload is the ground truth, the manifest is a
        # description of it, and this cache's history is exactly the kind that
        # makes them disagree.
        try:
            actual = len(torch.load(path))
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  shard {shard['file']} unreadable: {exc}")
            actual = -1

        if actual != shard["count"] or len(clip_ids) != shard["count"]:
            count_mismatches.append((shard_idx, shard["count"], actual))

        for offset, clip_id in enumerate(clip_ids):
            occurrences.setdefault(clip_id, []).append((shard_idx, offset))

    listed = {s["file"] for s in manifest["shards"]}
    on_disk = {
        f for f in os.listdir(cache_dir)
        if f.startswith(f"{split_name}_shard_") and f.endswith(".pt")
    }
    files_orphan = sorted(on_disk - listed)

    duplicates = {cid: locs for cid, locs in occurrences.items() if len(locs) > 1}

    # Do the duplicate copies actually agree?
    tensor_status: Dict[str, str] = {}
    if compare_tensors and duplicates:
        for clip_id, locs in duplicates.items():
            try:
                first_feats, first_label = get_shard(locs[0][0])[locs[0][1]]
            except Exception as exc:  # noqa: BLE001
                tensor_status[clip_id] = f"unreadable: {type(exc).__name__}"
                continue

            verdicts = set()
            for shard_idx, offset in locs[1:]:
                try:
                    feats, label = get_shard(shard_idx)[offset]
                except Exception as exc:  # noqa: BLE001
                    verdicts.add(f"unreadable: {type(exc).__name__}")
                    continue
                if label != first_label:
                    verdicts.add("LABEL MISMATCH")
                elif tuple(feats.shape) != tuple(first_feats.shape):
                    verdicts.add(
                        f"shape {tuple(first_feats.shape)} vs {tuple(feats.shape)}"
                    )
                elif torch.equal(feats, first_feats):
                    verdicts.add("identical")
                else:
                    verdicts.add("DIFFERENT VALUES")
            tensor_status[clip_id] = ", ".join(sorted(verdicts))

    failed_ids = set(manifest.get("failed_clip_ids") or [])
    cached_ids = set(occurrences)

    failed_but_cached = sorted(failed_ids & cached_ids)
    unexpected = sorted(cached_ids - target_ids)
    missing = sorted(target_ids - cached_ids - failed_ids)

    redundant_entries = sum(len(v) - 1 for v in duplicates.values())

    result = {
        "split": split_name,
        "expected": len(target_ids),
        "manifest_num_clips": manifest["num_clips"],
        "entries_total": sum(len(v) for v in occurrences.values()),
        "unique_cached": len(cached_ids),
        "duplicates": duplicates,
        "duplicate_tensor_status": tensor_status,
        "redundant_entries": redundant_entries,
        "failed_ids": sorted(failed_ids),
        "failed_but_cached": failed_but_cached,
        "unexpected_ids": unexpected,
        "missing_ids": missing,
        "count_mismatches": count_mismatches,
        "files_missing": files_missing,
        "files_orphan": files_orphan,
    }

    if verbose:
        _print_cache_audit(result)

    return result


def _print_cache_audit(a: Dict[str, Any]) -> None:
    print("=" * 64)
    print(f"CACHE AUDIT — {a['split']}")
    print("=" * 64)
    print(f"  expected (frozen split) : {a['expected']:,}")
    print(f"  manifest num_clips      : {a['manifest_num_clips']:,}")
    print(f"  entries across shards   : {a['entries_total']:,}")
    print(f"  unique clip IDs         : {a['unique_cached']:,}")
    print(f"  redundant entries       : {a['redundant_entries']:,}")
    print(f"  recorded failures       : {len(a['failed_ids']):,}")

    if a["duplicates"]:
        print(f"\n  DUPLICATED CLIP IDS ({len(a['duplicates'])}):")
        for clip_id, locs in sorted(a["duplicates"].items()):
            status = a["duplicate_tensor_status"].get(clip_id, "not compared")
            where = ", ".join(f"shard {s}@{o}" for s, o in locs)
            print(f"    {clip_id:<28} x{len(locs)}  [{status}]  {where}")

    if a["failed_but_cached"]:
        print(
            f"\n  STALE FAILURES ({len(a['failed_but_cached'])}) — listed as failed "
            "but present in a shard (recovered on a later resume):"
        )
        for cid in a["failed_but_cached"]:
            print(f"    {cid}")

    if a["unexpected_ids"]:
        print(f"\n  NOT IN THIS SPLIT ({len(a['unexpected_ids'])}):")
        for cid in a["unexpected_ids"][:20]:
            print(f"    {cid}")
        if len(a["unexpected_ids"]) > 20:
            print(f"    ... and {len(a['unexpected_ids']) - 20} more")

    if a["missing_ids"]:
        print(f"\n  MISSING ({len(a['missing_ids'])}) — neither cached nor failed:")
        for cid in a["missing_ids"][:20]:
            print(f"    {cid}")
        if len(a["missing_ids"]) > 20:
            print(f"    ... and {len(a['missing_ids']) - 20} more")

    if a["count_mismatches"]:
        print("\n  MANIFEST/PAYLOAD COUNT MISMATCHES (shard, manifest, actual):")
        for row in a["count_mismatches"]:
            print(f"    {row}")

    if a["files_missing"]:
        print(f"\n  SHARD FILES MISSING FROM DISK: {a['files_missing']}")
    if a["files_orphan"]:
        print(f"\n  SHARD FILES NOT IN MANIFEST: {a['files_orphan']}")

    # Preview assumes the default repair: dedup, drop cross-split clips, and
    # clear stale failure entries.
    kept = a["unique_cached"] - len(a["unexpected_ids"])
    live_failures = len(set(a["failed_ids"]) - set(a["failed_but_cached"]))
    reconciled = kept + live_failures
    print(
        f"\n  after repair would be: {kept:,} cached + {live_failures:,} failed "
        f"= {reconciled:,} vs {a['expected']:,} expected"
        + ("  OK" if reconciled == a["expected"] else "  STILL SHORT")
    )
    if reconciled != a["expected"] and a["missing_ids"]:
        print(
            f"  ({len(a['missing_ids']):,} clip(s) missing entirely — those need "
            "re-extraction, not repair)"
        )
    print("=" * 64)


def repair_feature_cache(
    split_name: str,
    cache_dir: str = FEATURE_CACHE_DIR,
    dry_run: bool = True,
    drop_unexpected: bool = True,
    target_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Deduplicate a cache and reconcile its manifest, without re-extracting.

    Actions, all idempotent:
      * keep the FIRST occurrence of each clip ID, drop later ones
      * optionally drop cached clips not in the split's target set
      * drop stale entries from failed_clip_ids for clips that are actually
        cached (recovered on a later resume)
      * rewrite only the shards that changed, then rewrite the manifest with
        corrected counts

    Run with dry_run=True first to see the plan. Nothing is deleted except
    redundant entries; feature data for kept clips is never rewritten in place
    beyond being re-saved into its own shard.
    """
    audit = audit_feature_cache(
        split_name, cache_dir, target_ids=target_ids, verbose=False
    )
    manifest = read_manifest(split_name, cache_dir)
    assert manifest is not None

    # (shard_idx, offset) pairs to remove.
    drop: Dict[int, set] = {}

    for clip_id, locs in audit["duplicates"].items():
        for shard_idx, offset in locs[1:]:          # keep locs[0]
            drop.setdefault(shard_idx, set()).add(offset)

    dropped_unexpected = 0
    if drop_unexpected and audit["unexpected_ids"]:
        unexpected = set(audit["unexpected_ids"])
        for shard_idx, shard in enumerate(manifest["shards"]):
            for offset, clip_id in enumerate(shard["clip_ids"]):
                if clip_id in unexpected:
                    drop.setdefault(shard_idx, set()).add(offset)
                    dropped_unexpected += 1

    stale_failures = set(audit["failed_but_cached"])
    new_failed = [
        cid for cid in (manifest.get("failed_clip_ids") or [])
        if cid not in stale_failures
    ]

    total_drops = sum(len(v) for v in drop.values())

    plan = {
        "split": split_name,
        "shards_to_rewrite": sorted(drop),
        "entries_to_drop": total_drops,
        "duplicate_entries_dropped": total_drops - dropped_unexpected,
        "unexpected_entries_dropped": dropped_unexpected,
        "stale_failures_cleared": sorted(stale_failures),
        "resulting_clips": audit["unique_cached"] - (
            len(audit["unexpected_ids"]) if drop_unexpected else 0
        ),
        "resulting_failed": len(new_failed),
        "expected": audit["expected"],
        "missing_ids": audit["missing_ids"],
    }

    print("\n--- REPAIR PLAN ---")
    print(f"  shards to rewrite       : {len(plan['shards_to_rewrite'])}")
    print(f"  duplicate entries drop  : {plan['duplicate_entries_dropped']}")
    print(f"  unexpected entries drop : {plan['unexpected_entries_dropped']}")
    print(f"  stale failures cleared  : {len(plan['stale_failures_cleared'])}")
    print(
        f"  result                  : {plan['resulting_clips']:,} cached + "
        f"{plan['resulting_failed']:,} failed = "
        f"{plan['resulting_clips'] + plan['resulting_failed']:,} "
        f"vs {plan['expected']:,} expected"
    )

    if dry_run:
        print("  DRY RUN — nothing written. Re-run with dry_run=False to apply.")
        return plan

    new_shards: List[Dict[str, Any]] = []
    for shard_idx, shard in enumerate(manifest["shards"]):
        path = os.path.join(cache_dir, shard["file"])

        if shard_idx not in drop:
            new_shards.append(shard)
            continue

        payload = torch.load(path)
        offsets = drop[shard_idx]

        kept_payload = [x for i, x in enumerate(payload) if i not in offsets]
        kept_ids = [
            c for i, c in enumerate(shard["clip_ids"]) if i not in offsets
        ]

        if not kept_payload:
            os.remove(path)
            print(f"  shard {shard['file']}: now empty, removed")
            continue

        tmp = path + ".tmp"
        torch.save(kept_payload, tmp)
        os.replace(tmp, path)

        new_shards.append({
            "file": shard["file"],
            "count": len(kept_payload),
            "bytes": os.path.getsize(path),
            "clip_ids": kept_ids,
        })
        print(
            f"  shard {shard['file']}: {shard['count']} -> {len(kept_payload)} "
            f"({len(offsets)} dropped)"
        )

    write_manifest(
        split_name,
        new_shards,
        manifest.get("store_dtype", "float16"),
        complete=True,
        cache_dir=cache_dir,
        capped=manifest.get("capped"),
        expected_clips=audit["expected"],
        failed_clip_ids=new_failed,
    )

    after = read_manifest(split_name, cache_dir) or {}
    print(
        f"\n  manifest rewritten: {after.get('num_clips'):,} cached + "
        f"{after.get('num_failed'):,} failed, "
        f"accounted_for={after.get('accounted_for')}"
    )

    plan["applied"] = True
    plan["accounted_for"] = after.get("accounted_for")
    return plan


def repair_all_caches(
    cache_dir: str = FEATURE_CACHE_DIR,
    splits: Sequence[str] = ("train", "validation"),
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Audit + repair every split, then re-verify. One call for the Kaggle cell."""
    out: Dict[str, Any] = {}
    for name in splits:
        if read_manifest(name, cache_dir) is None:
            print(f"[{name}] no manifest — skipping")
            continue
        audit_feature_cache(name, cache_dir)
        out[name] = repair_feature_cache(name, cache_dir, dry_run=dry_run)

    if not dry_run:
        print("\n" + "=" * 64)
        print("POST-REPAIR VERIFICATION")
        print("=" * 64)
        ok = True
        for name in splits:
            man = read_manifest(name, cache_dir)
            if man is None:
                continue
            good = man.get("accounted_for") is True
            ok = ok and good
            print(
                f"  [{name}] {man['num_clips']:,} + {man['num_failed']:,} = "
                f"{man['num_clips'] + man['num_failed']:,} / "
                f"{man['expected_clips']:,}  "
                f"accounted_for={man.get('accounted_for')}  "
                f"{'OK' if good else 'STILL BROKEN'}"
            )
        print(
            "\n"
            + ("All splits reconcile — run_step6_training will proceed."
               if ok else
               "Still not reconciling. Re-run audit_feature_cache for detail.")
        )
        out["all_ok"] = ok

    return out


class ShardedFeatureDataset(Dataset):
    """
    Reads a sharded feature cache without holding it all in RAM.

    Keeps at most `max_cached_shards` shards resident. Pair with
    ShardBlockShuffleSampler — plain shuffle=True would touch a different shard on
    almost every __getitem__ and thrash the cache badly.
    """

    def __init__(
        self,
        split_name: str,
        cache_dir: str = FEATURE_CACHE_DIR,
        max_cached_shards: int = 2,
        require_complete: bool = True,
    ):
        manifest = read_manifest(split_name, cache_dir)
        if manifest is None:
            raise FileNotFoundError(
                f"No feature cache manifest for '{split_name}' in {cache_dir}."
            )
        if require_complete and not manifest.get("complete", False):
            raise ValueError(
                f"Feature cache for '{split_name}' is incomplete "
                f"({manifest['num_clips']:,} clips). Finish the streaming pass, "
                "or pass require_complete=False to train on the partial cache."
            )

        self.cache_dir = cache_dir
        self.split_name = split_name
        self.manifest = manifest
        self.shards = manifest["shards"]
        self.max_cached_shards = max(1, max_cached_shards)

        # Global index -> (shard, offset), via cumulative shard start offsets.
        self.shard_starts: List[int] = []
        total = 0
        for shard in self.shards:
            self.shard_starts.append(total)
            total += shard["count"]
        self.total = total

        self._cache: Dict[int, List[Tuple[torch.Tensor, int]]] = {}
        self._order: List[int] = []

    def __len__(self) -> int:
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        import bisect

        shard_idx = bisect.bisect_right(self.shard_starts, idx) - 1
        return shard_idx, idx - self.shard_starts[shard_idx]

    def _get_shard(self, shard_idx: int) -> List[Tuple[torch.Tensor, int]]:
        if shard_idx in self._cache:
            return self._cache[shard_idx]

        path = os.path.join(self.cache_dir, self.shards[shard_idx]["file"])
        data = torch.load(path)

        self._cache[shard_idx] = data
        self._order.append(shard_idx)
        while len(self._order) > self.max_cached_shards:
            self._cache.pop(self._order.pop(0), None)

        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        shard_idx, offset = self._locate(idx)
        feats, label = self._get_shard(shard_idx)[offset]
        # Cast back to fp32 here so the model always sees the dtype it trained on.
        return feats.float(), float(label)

    def shard_bounds(self) -> List[Tuple[int, int]]:
        return [
            (start, start + shard["count"])
            for start, shard in zip(self.shard_starts, self.shards)
        ]


class ShardBlockShuffleSampler(Sampler):
    """
    Shuffles within a sliding window of shards instead of globally.

    Global shuffling over a sharded cache is pathological: each draw lands in a
    random shard, so every item costs a shard load. This shuffles the order of
    shard-groups and shuffles all indices inside a group, giving a shuffle window
    of window_shards * shard_size clips while keeping shard access sequential.

    window_shards must be <= the dataset's max_cached_shards, or the LRU evicts
    shards that the current window still needs.
    """

    def __init__(
        self,
        dataset: ShardedFeatureDataset,
        window_shards: int = 2,
        seed: int = SEED,
    ):
        self.bounds = dataset.shard_bounds()
        self.window_shards = max(1, window_shards)
        self.seed = seed
        self.epoch = 0
        self.total = len(dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total

    def __iter__(self) -> Iterator[int]:
        g = random.Random(self.seed + self.epoch)

        groups = [
            self.bounds[i:i + self.window_shards]
            for i in range(0, len(self.bounds), self.window_shards)
        ]
        g.shuffle(groups)

        for group in groups:
            indices = [i for start, end in group for i in range(start, end)]
            g.shuffle(indices)
            yield from indices


def load_sharded_cache_into_memory(
    split_name: str,
    cache_dir: str = FEATURE_CACHE_DIR,
    as_float32: bool = True,
) -> List[Tuple[torch.Tensor, int]]:
    """
    Load an entire sharded cache into one list.

    Only safe for splits that fit in RAM — validation (3,110 clips, ~1-3 GB) does;
    train (24,881 clips) generally does not. Use ShardedFeatureDataset for train.
    """
    manifest = read_manifest(split_name, cache_dir)
    if manifest is None:
        raise FileNotFoundError(f"No manifest for '{split_name}' in {cache_dir}.")

    out: List[Tuple[torch.Tensor, int]] = []
    for shard in manifest["shards"]:
        data = torch.load(os.path.join(cache_dir, shard["file"]))
        for feats, label in data:
            out.append((feats.float() if as_float32 else feats, label))
    return out


# ============================================================================
# SECTION 7 — AASIST-inspired classifier head
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Memory-efficient, NaN-safe graph attention.

    Scores are computed as separate src/dst projections that broadcast-add into
    (batch, seq, seq). The concat formulation would build (batch, seq, seq, 2*dim)
    and OOM on GPU — see module docstring.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.attn_src = nn.Linear(out_dim, 1)
        self.attn_dst = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.proj(x)

        src_scores = self.attn_src(h)
        dst_scores = self.attn_dst(h)
        e = src_scores + dst_scores.transpose(1, 2)
        e = self.leaky_relu(e)

        if mask is not None:
            pad_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            e = e.masked_fill(pad_mask == 0, float("-inf"))

        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # required: neutralizes NaN from fully-padded rows
        alpha = self.dropout(alpha)

        out = torch.bmm(alpha, h)

        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return out


class AASISTHead(nn.Module):
    """
    Trained classifier head over frozen frontend features.

    Emits a single logit per clip. Higher logit = more fake, since training targets
    are the metadata labels (1 = fake). See SCORE ORIENTATION in Section 9.
    """

    def __init__(
        self,
        input_dim: int = FRONTEND_DIM,
        hidden_dim: int = 256,
        num_gat_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_gat_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pool_attn = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(x)

        for gat in self.gat_layers:
            residual = h
            h = gat(h, mask=mask)
            h = self.layer_norm(h + residual)
            h = self.dropout(h)
            if mask is not None:
                h = h * mask.unsqueeze(-1)

        attn_scores = self.pool_attn(h).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)

        pooled = (h * attn_weights).sum(dim=1)
        logit = self.classifier(pooled).squeeze(-1)
        return logit


def build_model(config: Optional[TrainConfig] = None) -> AASISTHead:
    config = config or TrainConfig()
    return AASISTHead(
        input_dim=FRONTEND_DIM,
        hidden_dim=config.hidden_dim,
        num_gat_layers=config.num_gat_layers,
        dropout=config.dropout,
    ).to(DEVICE)


# ============================================================================
# SECTION 8 — Dataset and padded-batch collation
# ============================================================================

class CachedFeatureDataset(Dataset):
    """Wraps a list of (feat_tensor, label) tuples from the feature cache."""

    def __init__(self, cached_features: List[Tuple[torch.Tensor, int]]):
        self.data = cached_features

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        feats, label = self.data[idx]
        return feats, float(label)


def collate_fn(batch):
    """Right-pad variable-length feature sequences and emit a float frame mask."""
    feats_list, labels = zip(*batch)

    lengths = [f.shape[0] for f in feats_list]
    max_len = max(lengths)
    feat_dim = feats_list[0].shape[1]

    padded = torch.zeros(len(batch), max_len, feat_dim)
    mask = torch.zeros(len(batch), max_len)

    for i, f in enumerate(feats_list):
        L = f.shape[0]
        padded[i, :L, :] = f
        mask[i, :L] = 1

    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, mask, labels


def make_dataloader(
    features_or_dataset,
    batch_size: int = 16,
    shuffle: bool = False,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    """
    Accepts either an in-memory [(feats, label), ...] list or a Dataset
    (e.g. ShardedFeatureDataset).

    DataLoader rejects shuffle=True together with a sampler, so a supplied sampler
    takes precedence.
    """
    dataset = (
        features_or_dataset
        if isinstance(features_or_dataset, Dataset)
        else CachedFeatureDataset(features_or_dataset)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
    )


# ============================================================================
# SECTION 9 — Metrics
# ============================================================================
# SCORE ORIENTATION — read before wiring anything into compute_eer.
#
# compute_eer() labels its first argument as the positive class, so it expects
# *bonafide-ness* scores: higher = more likely real.
#
# The model emits *fake-ness* logits (trained against 1 = fake).
#
# The two are opposite. Feeding raw logits in as bonafide scores yields an EER of
# roughly (1 - true EER) — a strong model reports ~94% instead of ~6%. Always route
# through bonafide_scores_from_logits() rather than passing logits directly.

def bonafide_scores_from_logits(logits: np.ndarray) -> np.ndarray:
    """Convert fake-ness logits to bonafide-ness scores for compute_eer()."""
    return -np.asarray(logits, dtype=np.float64)


def compute_eer(bonafide_scores, spoof_scores) -> Tuple[float, float]:
    """
    Equal Error Rate. Returns (eer, threshold).

    Sanity-checked: perfect classifier -> 0%, random -> ~50%, two unit-variance
    Gaussians +/-1 apart -> ~15.87% theoretical, matched empirically.
    """
    from sklearn.metrics import roc_curve

    labels = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer, thresholds[eer_idx]


def eer_from_logits_and_labels(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    Convenience wrapper that applies the score-orientation flip correctly.

    `labels` uses the metadata convention (0 = real, 1 = fake).
    """
    scores = bonafide_scores_from_logits(logits)
    labels = np.asarray(labels)
    return compute_eer(scores[labels == LABEL_REAL], scores[labels == LABEL_FAKE])


class TemperatureScaler(nn.Module):
    """Single-parameter post-hoc calibration. Fit on validation logits only."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100) -> float:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Expected Calibration Error. Returns (ece, per_bin_data)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            acc_in_bin = labels[in_bin].mean()
            conf_in_bin = probs[in_bin].mean()
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
            bin_data.append((lo, hi, conf_in_bin, acc_in_bin, prop_in_bin))
        else:
            bin_data.append((lo, hi, None, None, 0))

    return ece, bin_data


# ============================================================================
# SECTION 10 — Training
# ============================================================================
# Reconciled against the verified Step 5 loop (Colab "Cell part 4"/"Cell part 5",
# loss 0.7146 -> ~0.09 over 15 epochs on a 150-clip subset, checkpoint reload
# confirmed). The inner loop below is step-for-step identical to that code.
#
# The verified run had NO LR scheduler, NO early stopping, and NO gradient
# clipping. Anything here beyond that is marked [beyond verified] at its site.

def build_optimizer(model: AASISTHead, config: TrainConfig) -> torch.optim.Optimizer:
    """
    AdamW over head parameters.

    The verified run passed head.parameters() directly. The requires_grad filter
    here is a no-op for the head (every head parameter is trainable) and exists
    only to guard against frontend parameters ever being handed in by mistake.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_one_epoch(
    model: AASISTHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float = 0.0,
) -> float:
    """
    One training epoch. Returns the mean of the per-batch losses.

    model.train() is called here rather than once before the epoch loop (as in the
    verified cell) because train_model() runs a validation pass between epochs,
    and evaluate() puts the model in eval mode. Without this call, epochs 2+ would
    train with dropout disabled. This preserves the verified semantics under
    interleaved validation rather than departing from them.
    """
    model.train()
    epoch_losses: List[float] = []

    for batch_feats, batch_mask, batch_labels in loader:
        batch_feats = batch_feats.to(DEVICE)
        batch_mask = batch_mask.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch_feats, mask=batch_mask)
        loss = criterion(logits, batch_labels)
        loss.backward()

        # [beyond verified] Disabled by default (grad_clip=0.0) to match the run.
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        epoch_losses.append(loss.item())

    return sum(epoch_losses) / max(len(epoch_losses), 1)


@torch.no_grad()
def collect_logits(
    model: AASISTHead,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference over a loader. Returns (logits, labels) as numpy arrays."""
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for feats, mask, labels in loader:
        feats = feats.to(DEVICE)
        mask = mask.to(DEVICE)

        logits = model(feats, mask=mask)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels)


def evaluate(
    model: AASISTHead,
    loader: DataLoader,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Validation/test pass. Returns loss and EER."""
    logits, labels = collect_logits(model, loader)

    eer, threshold = eer_from_logits_and_labels(logits, labels)

    result = {"eer": float(eer), "threshold": float(threshold)}

    if criterion is not None:
        loss = criterion(
            torch.from_numpy(logits).float(),
            torch.from_numpy(labels).float(),
        )
        result["loss"] = float(loss.item())

    return result


def save_checkpoint(
    model: AASISTHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_history: List[float],
    path: str,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[TrainConfig] = None,
    best_eer: Optional[float] = None,
    history: Optional[List[Dict[str, float]]] = None,
) -> str:
    """
    Head weights only — the frontend is frozen and reloaded from HF by name.

    The first four keys are the verified checkpoint format and must stay exactly
    as-is; Step 5 confirmed save/reload against them:

        model_state_dict, optimizer_state_dict, epoch, loss_history

    metrics/config/frontend_model_name are [beyond verified] additions. They are
    additive only — nothing reading the four verified keys is affected — and
    `config` is what lets load_checkpoint() rebuild the head without being told
    its shape. Drop them if the checkpoint format becomes a shared contract.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss_history": loss_history,
    }

    if metrics is not None:
        payload["metrics"] = metrics
    if config is not None:
        payload["config"] = config.__dict__
        payload["frontend_model_name"] = FRONTEND_MODEL_NAME
    if best_eer is not None:
        # Required for correct resume. Without it a resumed run restarts
        # best-tracking at infinity and overwrites best_model.pt with the first
        # epoch it sees, even when that epoch is worse than the pre-crash best.
        payload["best_eer"] = best_eer
    if history is not None:
        payload["history"] = history

    # Write to a temp file and rename, so a disconnect mid-write cannot leave a
    # truncated checkpoint where a valid one used to be.
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str,
    model: Optional[AASISTHead] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Restore a checkpoint. Builds the model from stored config if not supplied."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=DEVICE)

    if model is None:
        if "config" not in ckpt:
            raise ValueError(
                f"Checkpoint {path} carries no 'config' key, so the head shape "
                "cannot be inferred. Pass an explicit model= built with the same "
                "hyperparameters used to train it."
            )
        model = build_model(TrainConfig(**ckpt["config"]))

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    ckpt["model"] = model
    return ckpt


def train_model(
    train_features,
    val_features=None,
    config: Optional[TrainConfig] = None,
    checkpoint_dir: str = CHECKPOINT_DIR,
    validate: bool = True,
    train_sampler: Optional[Sampler] = None,
    resume: bool = True,
) -> Tuple[AASISTHead, Dict[str, Any]]:
    """
    Train the head on cached frozen features.

    The optimizer/criterion/inner loop match the verified Step 5 cell exactly.
    Two things here are [beyond verified], because the smoke test had no
    validation split at all:

      * a validation pass after each epoch, and
      * best-checkpoint selection on validation EER.

    Selection is not early stopping — training always runs the full num_epochs.
    Pass validate=False to reproduce the verified train-only behaviour.
    """
    config = config or TrainConfig()
    set_seed(config.seed)

    model = build_model(config)
    optimizer = build_optimizer(model, config)
    criterion = nn.BCEWithLogitsLoss()

    if train_sampler is not None:
        train_loader = make_dataloader(
            train_features, config.batch_size, sampler=train_sampler
        )
    else:
        train_loader = make_dataloader(
            train_features, config.batch_size, shuffle=config.shuffle_train
        )

    val_loader = (
        make_dataloader(val_features, config.batch_size, shuffle=False)
        if validate and val_features is not None and len(val_features)
        else None
    )

    loss_history: List[float] = []   # per-epoch mean loss — verified format
    history: List[Dict[str, float]] = []
    best_eer = float("inf")
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    last_path = os.path.join(checkpoint_dir, "last_model.pt")

    # --- resume from the last completed epoch -------------------------------
    start_epoch = 1
    if resume and os.path.exists(last_path):
        ckpt = load_checkpoint(last_path, model=model, optimizer=optimizer)

        stored = ckpt.get("config") or {}
        for key in ("hidden_dim", "num_gat_layers", "dropout"):
            if key in stored and stored[key] != getattr(config, key):
                raise ValueError(
                    f"Checkpoint {last_path} was trained with {key}="
                    f"{stored[key]}, but this run uses {getattr(config, key)}. "
                    "Resuming would mix two different models. Delete the "
                    "checkpoint to start fresh, or pass a matching TrainConfig."
                )

        completed = int(ckpt["epoch"])
        start_epoch = completed + 1
        loss_history = list(ckpt.get("loss_history", []))
        history = list(ckpt.get("history", []))
        best_eer = float(ckpt.get("best_eer", float("inf")))

        if len(loss_history) != completed:
            print(
                f"  WARNING: checkpoint says epoch {completed} but carries "
                f"{len(loss_history)} loss entries; trusting the epoch number."
            )

        print(
            f"Resuming from {last_path}: {completed} epoch(s) done, "
            f"continuing at epoch {start_epoch}/{config.num_epochs}"
            + (
                f", best val EER so far {best_eer * 100:.2f}%"
                if best_eer != float("inf") else ", no best recorded yet"
            )
        )
        # Note: RNG state is not restored, so a resumed run is not bit-identical
        # to an uninterrupted one. Per-epoch shuffling stays deterministic
        # because the sampler is seeded by epoch number, not by stream position.

        if start_epoch > config.num_epochs:
            print(
                f"Training already complete ({completed}/{config.num_epochs} "
                "epochs). Nothing to do."
            )
            return model, {
                "history": history,
                "loss_history": loss_history,
                "best_eer": best_eer if best_eer != float("inf") else None,
                "best_path": best_path if os.path.exists(best_path) else None,
                "last_path": last_path,
                "resumed": True,
                "epochs_run": 0,
            }

    for epoch in range(start_epoch, config.num_epochs + 1):
        # Without this the shard-block sampler replays one fixed order every epoch.
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.grad_clip
        )
        loss_history.append(avg_loss)

        record: Dict[str, float] = {"epoch": epoch, "train_loss": avg_loss}
        line = f"Epoch {epoch}/{config.num_epochs} | avg loss: {avg_loss:.4f}"

        val_metrics: Optional[Dict[str, float]] = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion)
            record["val_loss"] = val_metrics.get("loss", float("nan"))
            record["val_eer"] = val_metrics["eer"]
            line += (
                f" | val loss: {record['val_loss']:.4f}"
                f" | val EER: {val_metrics['eer'] * 100:.2f}%"
            )

        history.append(record)
        print(line)

        # Written every epoch, improvement or not, so a disconnect costs at most
        # one epoch. best_eer rides along so a resume does not lose best-tracking.
        save_checkpoint(
            model, optimizer, epoch, loss_history, last_path,
            metrics=val_metrics, config=config,
            best_eer=best_eer if best_eer != float("inf") else None,
            history=history,
        )

        if val_metrics is not None and val_metrics["eer"] < best_eer:
            best_eer = val_metrics["eer"]
            save_checkpoint(
                model, optimizer, epoch, loss_history, best_path,
                metrics=val_metrics, config=config,
                best_eer=best_eer, history=history,
            )
            # Re-write last_model.pt so its best_eer matches; otherwise a resume
            # right after a new best would forget that this epoch was the best.
            save_checkpoint(
                model, optimizer, epoch, loss_history, last_path,
                metrics=val_metrics, config=config,
                best_eer=best_eer, history=history,
            )
            print(f"  new best val EER {best_eer * 100:.2f}% -> {best_path}")

    print("\nLoss history:", [round(l, 4) for l in loss_history])
    print("Loss decreased overall:", loss_history[-1] < loss_history[0])

    return model, {
        "history": history,
        "loss_history": loss_history,
        "best_eer": best_eer if val_loader is not None else None,
        "best_path": best_path if val_loader is not None else None,
        "last_path": last_path,
        "resumed": start_epoch > 1,
        "epochs_run": config.num_epochs - start_epoch + 1,
    }


# ============================================================================
# SECTION 11 — Shared interface contract (Ashitha: PDF report generator + UI)
# ============================================================================
# PLACEHOLDER. The call signatures below are settled:
#
#     analyze_bytes(audio_bytes) -> AnalysisResult
#     evaluate_dataset(clip_ids, labels) -> dict
#
# The AnalysisResult dataclass fields and the evaluate_dataset return schema are
# still TBD — Ashitha is building the report generator and UI against them, so the
# exact shape must come from her before this is filled in. Do not guess the fields;
# a mismatch breaks her side silently.
#
# Record any change to this section in docs/TEAM_SYNC_LOG.md under
# "Interface changes".


@dataclass
class AnalysisResult:
    """
    PLACEHOLDER — fields TBD, pending Ashitha's exact contract.

    Likely to include some form of: verdict/label, confidence or probability,
    calibrated score, and the decision threshold used. Left deliberately empty
    rather than guessed.
    """

    # TODO(interface): replace with Ashitha's agreed fields.
    raw: Dict[str, Any] = field(default_factory=dict)


def analyze_bytes(audio_bytes: bytes) -> AnalysisResult:
    """
    PLACEHOLDER — single-clip inference from raw audio bytes.

    Intended flow once the contract lands:
        1. decode bytes -> waveform, sample_rate
        2. ensure_target_sample_rate(...)
        3. extract_features(...)
        4. load best checkpoint head, forward a batch of 1 (mask of ones)
        5. apply the fitted TemperatureScaler
        6. threshold at the validation-selected operating point
        7. populate AnalysisResult
    """
    raise NotImplementedError(
        "analyze_bytes is pending the shared interface contract with Ashitha."
    )


def evaluate_dataset(clip_ids: Sequence[str], labels: Sequence[int]) -> dict:
    """
    PLACEHOLDER — batch evaluation over a set of clip IDs.

    Intended flow once the contract lands:
        1. resolve features for clip_ids (feature cache first, stream as fallback)
        2. collect_logits over the batch
        3. eer_from_logits_and_labels(...) for overall EER
        4. per-language breakdown via build_label_lookup()'s language map
        5. calibrated probabilities -> compute_ece(...)
        6. return the agreed dict schema

    Per-language caveat to carry into the report: Gujarati has 156 test clips vs
    200 for the majority languages (Tamil 191, Odia 182, Bengali 180), so its
    per-language EER has a materially wider confidence interval.
    """
    raise NotImplementedError(
        "evaluate_dataset is pending the shared interface contract with Ashitha."
    )


# ============================================================================
# SECTION 12 — Step 6: full training run
# ============================================================================
# End-to-end training + validation-based model selection. Produces the first real
# EER numbers (per TEAM_SYNC_LOG, nothing is reportable before this runs).
#
# The test split is deliberately NOT touched here. Step 6 is training plus
# validation-based selection; the test pass is a separate, later step so the test
# set stays untouched while hyperparameters are still moving.
#
# Scope this cycle: 1 seed (SEED = 42), reduced robustness matrix, no ONNX
# quantization.


def run_step6_training(
    config: Optional[TrainConfig] = None,
    cache_dir: str = FEATURE_CACHE_DIR,
    checkpoint_dir: str = CHECKPOINT_DIR,
    shard_size: int = 500,
    store_dtype: torch.dtype = torch.float16,
    train_in_memory: bool = False,
    window_shards: int = 4,
    max_clips_per_split: Optional[int] = None,
    skip_extraction: bool = False,
    extract_batch_size: int = 8,
    max_failure_pct: float = 1.0,
) -> Dict[str, Any]:
    """
    Full Step 6 run: guards -> feature cache -> train -> calibrate.

    Stages:
        1. assert_split_integrity()   — refuse to run on a corrupted split
        2. verify_frontend_frozen()   — assert 0 trainable frontend params
        3. load frozen split + label lookup
        4. build_feature_caches_streaming() for train + validation in ONE pass
           (skipped if already complete, or if skip_extraction=True)
        5. train_model() over the sharded cache, checkpointing every epoch
        6. fit TemperatureScaler on VALIDATION logits only, report EER + ECE

    Args:
        train_in_memory: load the whole train cache into RAM. Faster, but needs
            ~8-25 GB depending on clip length and store_dtype — see the MEMORY
            note in Section 6. Default False uses the shard-backed dataset.
        max_clips_per_split: cap per split. Use a small value (e.g. 200) for a
            cheap end-to-end rehearsal before committing to the full ~28k clips.
        skip_extraction: reuse an existing complete cache and go straight to
            training.

    Returns a summary dict; also leaves checkpoints in checkpoint_dir.
    """
    config = config or TrainConfig()
    set_seed(config.seed)

    print("=" * 60)
    print("STEP 6 — FULL TRAINING RUN")
    print("=" * 60)

    # --- 1-2. guards --------------------------------------------------------
    print("\n[1/6] Verifying frozen split integrity...")
    assert_split_integrity()
    print("  split OK — 31,102 clips, no duplicates, no cross-split overlap.")

    print("\n[2/6] Verifying frontend is frozen...")
    n_trainable = verify_frontend_frozen()
    print(f"  frontend trainable params: {n_trainable}")

    # --- 3. split + labels --------------------------------------------------
    print("\n[3/6] Loading frozen split and verified metadata...")
    train_ids, val_ids, _test_ids = load_frozen_split_sets()
    label_lookup, language_lookup = build_label_lookup()
    print(f"  train {len(train_ids):,} | validation {len(val_ids):,}")
    print("  test split intentionally untouched in Step 6.")

    # --- 4. feature cache ---------------------------------------------------
    targets = {"train": set(train_ids), "validation": set(val_ids)}

    if skip_extraction:
        print("\n[4/6] skip_extraction=True — reusing existing cache.")
        manifests = {
            name: read_manifest(name, cache_dir) or {}
            for name in targets
        }
        for name, man in manifests.items():
            if not man or not man.get("complete"):
                raise ValueError(
                    f"skip_extraction=True but the '{name}' cache is missing or "
                    "incomplete. Run without skip_extraction first."
                )
    else:
        print("\n[4/6] Building feature cache (single streaming pass)...")
        manifests = build_feature_caches_streaming(
            targets=targets,
            label_lookup=label_lookup,
            language_lookup=language_lookup,
            cache_dir=cache_dir,
            shard_size=shard_size,
            store_dtype=store_dtype,
            resume=True,
            max_clips_per_split=max_clips_per_split,
            extract_batch_size=extract_batch_size,
        )

    for name, man in manifests.items():
        if not man.get("complete"):
            raise RuntimeError(
                f"Feature cache for '{name}' is incomplete. Re-run to resume "
                "(resume=True skips clips already extracted)."
            )

        # A rehearsal cache is internally complete but is not the full split.
        # Without this guard, `skip_extraction=True` after a capped rehearsal
        # would train on a few hundred clips and report the EER as if it were real.
        cap = man.get("capped")
        if cap is not None and max_clips_per_split is None:
            raise RuntimeError(
                f"The '{name}' cache was built with max_clips_per_split={cap} "
                f"({man['num_clips']:,} clips) — it is a rehearsal cache, not the "
                f"full split ({len(targets[name]):,} clips). Delete {cache_dir} and "
                "rebuild, or pass the same max_clips_per_split to acknowledge it."
            )
        if cap is not None:
            print(
                f"  [{name}] REHEARSAL CACHE — {man['num_clips']:,} clips "
                f"(cap {cap}). Results are not reportable."
            )

        # Surface any shortfall explicitly. A cache is "complete" once every clip
        # is accounted for, which includes clips that failed to decode — so the
        # only way a shortfall becomes visible is by checking it here.
        n_failed = man.get("num_failed", 0)
        expected = man.get("expected_clips")
        if n_failed:
            pct = n_failed / expected * 100 if expected else 0.0
            print(
                f"  [{name}] {man['num_clips']:,} cached, {n_failed:,} failed to "
                f"decode ({pct:.2f}% of {expected:,})"
            )
            if man.get("accounted_for") is False:
                raise RuntimeError(
                    f"'{name}' cache does not add up: {man['num_clips']:,} cached "
                    f"+ {n_failed:,} failed != {expected:,} expected. Refusing to "
                    "train on a cache whose contents cannot be accounted for."
                )
            if pct > max_failure_pct:
                raise RuntimeError(
                    f"'{name}' lost {pct:.2f}% of clips to decode failures, above "
                    f"max_failure_pct={max_failure_pct}%. Investigate "
                    f"{failed_clips_path(cache_dir)} before training; re-running "
                    "with resume=True retries failed clips."
                )

    # --- 5. training --------------------------------------------------------
    print("\n[5/6] Preparing datasets...")

    # Validation is small enough to hold in RAM, and doing so avoids reloading
    # shards on every epoch's validation pass.
    val_data = load_sharded_cache_into_memory("validation", cache_dir)
    print(f"  validation: {len(val_data):,} clips in memory")

    train_sampler = None
    if train_in_memory:
        train_data = load_sharded_cache_into_memory("train", cache_dir)
        print(f"  train: {len(train_data):,} clips in memory")
    else:
        train_data = ShardedFeatureDataset(
            "train",
            cache_dir=cache_dir,
            max_cached_shards=max(window_shards, 2),
        )
        train_sampler = ShardBlockShuffleSampler(
            train_data, window_shards=window_shards, seed=config.seed
        )
        print(
            f"  train: {len(train_data):,} clips across "
            f"{len(train_data.shards)} shards (shard-backed), "
            f"shuffle window {window_shards * shard_size:,} clips"
        )

    print("\n[5/6] Training...")
    model, train_summary = train_model(
        train_features=train_data,
        val_features=val_data,
        config=config,
        checkpoint_dir=checkpoint_dir,
        validate=True,
        train_sampler=train_sampler,
    )

    # --- 6. calibration on validation only -----------------------------------
    print("\n[6/6] Calibrating on validation logits...")

    best_path = train_summary.get("best_path")
    if best_path and os.path.exists(best_path):
        ckpt = load_checkpoint(best_path, model=model)
        print(f"  loaded best checkpoint from epoch {ckpt['epoch']}")

    val_loader = make_dataloader(val_data, config.batch_size, shuffle=False)
    val_logits, val_labels = collect_logits(model, val_loader)

    uncalibrated_eer, threshold = eer_from_logits_and_labels(val_logits, val_labels)

    scaler = TemperatureScaler().to(DEVICE)
    temperature = scaler.fit(
        torch.from_numpy(val_logits).float().to(DEVICE),
        torch.from_numpy(val_labels).float().to(DEVICE),
    )

    with torch.no_grad():
        calibrated_logits = (
            scaler(torch.from_numpy(val_logits).float().to(DEVICE)).cpu().numpy()
        )

    # ECE is computed on P(fake), matching the training target (1 = fake).
    probs_raw = 1.0 / (1.0 + np.exp(-val_logits))
    probs_cal = 1.0 / (1.0 + np.exp(-calibrated_logits))
    ece_raw, _ = compute_ece(probs_raw, val_labels)
    ece_cal, _ = compute_ece(probs_cal, val_labels)

    # Temperature scaling is monotonic, so EER is unchanged by construction —
    # recomputed here only as a guard against a sign/orientation regression.
    calibrated_eer, _ = eer_from_logits_and_labels(calibrated_logits, val_labels)

    print(f"  validation EER      : {uncalibrated_eer * 100:.2f}%")
    print(f"  temperature         : {temperature:.4f}")
    print(f"  ECE before          : {ece_raw:.4f}")
    print(f"  ECE after           : {ece_cal:.4f}")
    print(f"  EER after scaling   : {calibrated_eer * 100:.2f}% (should be unchanged)")

    summary = {
        "config": config.__dict__,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "loss_history": train_summary["loss_history"],
        "history": train_summary["history"],
        "best_val_eer": train_summary["best_eer"],
        "best_checkpoint": train_summary["best_path"],
        "val_eer": float(uncalibrated_eer),
        "val_eer_threshold": float(threshold),
        "temperature": float(temperature),
        "ece_before": float(ece_raw),
        "ece_after": float(ece_cal),
    }

    results_path = os.path.join(checkpoint_dir, "step6_summary.json")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {results_path}")
    print("Remember to record the EER in docs/TEAM_SYNC_LOG.md.")

    return summary


# ============================================================================
# SECTION 13 — Resilient Colab entry point
# ============================================================================
# Everything below exists so that ONE cell, re-run after ANY disconnect at ANY
# point, does the right thing: skip what is done, resume what is partial,
# continue. Correctness and idempotency over elegance.
#
# The single most important part is that state lives on Google Drive. Colab's
# local disk is wiped on disconnect, so a feature cache or checkpoint under
# /content is gone exactly when resume matters most.

DRIVE_MOUNT_POINT = "/content/drive"
DEFAULT_DRIVE_WORKDIR = "/content/drive/MyDrive/indic-fake-voice-detection"
REPO_URL = "https://github.com/Shruti-sivaramakrishnan/indic-fake-voice-detection.git"
DEFAULT_REPO_DIR = "/content/indic-fake-voice-detection"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def ensure_drive_mounted(mount_point: str = DRIVE_MOUNT_POINT) -> bool:
    """
    Mount Google Drive, idempotently. Returns True if usable.

    Never assumes an existing mount: a stale /content/drive directory can survive
    a disconnect while the mount itself is dead.
    """
    if not in_colab():
        print("Not running in Colab — skipping Drive mount.")
        return False

    my_drive = os.path.join(mount_point, "MyDrive")
    if os.path.isdir(my_drive) and os.access(my_drive, os.W_OK):
        print(f"Drive already mounted at {mount_point}")
        return True

    from google.colab import drive  # type: ignore

    for attempt in (1, 2):
        try:
            print(f"Mounting Google Drive (attempt {attempt})...")
            drive.mount(mount_point, force_remount=(attempt > 1))
            if os.path.isdir(my_drive):
                print(f"Drive mounted at {mount_point}")
                return True
        except Exception as exc:  # noqa: BLE001
            print(f"  mount attempt {attempt} failed: {type(exc).__name__}: {exc}")

    print(
        "\n" + "!" * 60
        + "\nCOULD NOT MOUNT GOOGLE DRIVE."
        + "\nAuthorize the mount popup and re-run this cell. Without Drive, the"
        + "\nfeature cache and checkpoints live on Colab's local disk and are"
        + "\nERASED on the next disconnect.\n"
        + "!" * 60 + "\n"
    )
    return False


def ensure_repo(
    repo_dir: str = DEFAULT_REPO_DIR,
    repo_url: str = REPO_URL,
    branch: str = "claude/ml-repo-structure-review-fpox8u",
) -> str:
    """Clone the repo if absent, otherwise fast-forward it. Idempotent."""
    import subprocess

    marker = os.path.join(repo_dir, "model_pipeline.py")

    if os.path.exists(marker):
        print(f"Repo present at {repo_dir} — pulling latest...")
        subprocess.run(
            ["git", "-C", repo_dir, "pull", "--ff-only", "origin", branch],
            check=False,
        )
        return repo_dir

    print(f"Cloning {repo_url} ({branch}) -> {repo_dir}")
    subprocess.run(
        ["git", "clone", "--branch", branch, repo_url, repo_dir], check=True
    )
    return repo_dir


def _fmt_age(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    age = time.time() - os.path.getmtime(path)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    return f"{stamp} ({_format_duration(age)} ago)"


def print_status_report(
    cache_dir: str = FEATURE_CACHE_DIR,
    checkpoint_dir: str = CHECKPOINT_DIR,
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """
    One glance at where the run stands. Reads only on-disk state, so it is
    accurate regardless of what this Python session has or has not done.
    """
    config = config or TrainConfig()

    print("=" * 64)
    print("STEP 6 STATUS")
    print("=" * 64)
    print(f"cache dir      : {cache_dir}")
    print(f"checkpoint dir : {checkpoint_dir}")

    status: Dict[str, Any] = {"cache": {}, "training": {}}

    print("\nFEATURE CACHE")
    try:
        train_ids, val_ids, _ = load_frozen_split_sets()
        expected = {"train": len(train_ids), "validation": len(val_ids)}
    except Exception:  # noqa: BLE001
        expected = {}

    for name in ("train", "validation"):
        man = read_manifest(name, cache_dir)
        exp = man.get("expected_clips") if man else None
        exp = exp or expected.get(name)
        if not man:
            print(f"  {name:<11}: not started" + (f" (0 / {exp:,})" if exp else ""))
            status["cache"][name] = {"pct": 0.0, "complete": False, "clips": 0}
            continue

        clips = man["num_clips"]
        failed = man.get("num_failed", 0)
        pct = (clips + failed) / exp * 100 if exp else 0.0
        flag = "COMPLETE" if man.get("complete") else "partial"
        cap = man.get("capped")
        print(
            f"  {name:<11}: {clips:,} cached"
            + (f", {failed:,} failed" if failed else "")
            + (f" / {exp:,} expected" if exp else "")
            + f"  ->  {pct:5.1f}% {flag}"
            + (f"  [REHEARSAL cap={cap}]" if cap else "")
        )
        status["cache"][name] = {
            "pct": pct, "complete": bool(man.get("complete")), "clips": clips,
            "failed": failed, "capped": cap,
        }

    failures = read_failed_clips(cache_dir)
    if failures:
        print(f"  failure log: {len(failures)} event(s) — {failed_clips_path(cache_dir)}")

    print("\nTRAINING")
    last_path = os.path.join(checkpoint_dir, "last_model.pt")
    best_path = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(last_path):
        print(f"  no checkpoint — training has not started (0 / {config.num_epochs})")
        status["training"] = {"epochs_done": 0, "complete": False}
    else:
        try:
            ckpt = torch.load(last_path, map_location="cpu")
            done = int(ckpt.get("epoch", 0))
            best = ckpt.get("best_eer")
            print(
                f"  last_model.pt : epoch {done} of {config.num_epochs} completed"
                f"  [{_fmt_age(last_path)}]"
            )
            if best is not None:
                print(f"  best val EER  : {best * 100:.2f}%  [{_fmt_age(best_path)}]")
            else:
                print(f"  best_model.pt : {_fmt_age(best_path)}")
            lh = ckpt.get("loss_history") or []
            if lh:
                print(
                    f"  loss          : {lh[0]:.4f} -> {lh[-1]:.4f} "
                    f"over {len(lh)} epoch(s)"
                )
            complete = done >= config.num_epochs
            print(f"  status        : {'COMPLETE' if complete else 'PARTIAL'}")
            status["training"] = {
                "epochs_done": done, "complete": complete, "best_eer": best,
            }
        except Exception as exc:  # noqa: BLE001
            print(f"  checkpoint unreadable ({type(exc).__name__}: {exc})")
            status["training"] = {"epochs_done": 0, "complete": False, "error": str(exc)}

    summary_path = os.path.join(checkpoint_dir, "step6_summary.json")
    if os.path.exists(summary_path):
        print(f"\nSUMMARY EXISTS : {summary_path}  [{_fmt_age(summary_path)}]")
    print("=" * 64 + "\n")

    return status


def write_git_ready_summary(
    summary: Dict[str, Any],
    repo_dir: str,
    checkpoint_dir: str,
) -> str:
    """
    Write docs/STEP6_RESULTS.md the moment training + calibration finish, so the
    next action is a commit and nothing is ambiguous about whether it is done.
    """
    docs_dir = os.path.join(repo_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, "STEP6_RESULTS.md")

    cfg = summary.get("config", {})
    lh = summary.get("loss_history") or []

    lines = [
        "# Step 6 Results — Model Pipeline",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
        "`model_pipeline.run_step6_resilient()`._",
        "",
        "## Headline",
        "",
        f"- **Validation EER: {summary['val_eer'] * 100:.2f}%**",
        f"- Best checkpoint val EER: "
        + (
            f"{summary['best_val_eer'] * 100:.2f}%"
            if summary.get("best_val_eer") is not None else "n/a"
        ),
        f"- ECE before calibration: {summary['ece_before']:.4f}",
        f"- ECE after calibration: {summary['ece_after']:.4f}",
        f"- Temperature: {summary['temperature']:.4f}",
        f"- EER decision threshold: {summary['val_eer_threshold']:.4f}",
        "",
        "## Data",
        "",
        f"- Train clips: {summary['num_train']:,}",
        f"- Validation clips: {summary['num_val']:,}",
        "- Split: frozen `data_pipeline/split_ids.json` (31,102 clips), loaded "
        "as-is, never regenerated.",
        "- Test split untouched — Step 6 is training + validation-based selection "
        "only.",
        "",
        "## Configuration",
        "",
        "```",
    ]
    for key in sorted(cfg):
        lines.append(f"{key} = {cfg[key]}")
    lines += [
        "```",
        "",
        f"- Frontend: frozen `{FRONTEND_MODEL_NAME}` (0 trainable parameters)",
        f"- Seed: {cfg.get('seed', SEED)} (1 seed this cycle — compressed timeline)",
        "",
        "## Training curve",
        "",
        f"- Epochs completed: {len(lh)}",
    ]
    if lh:
        lines.append(f"- Loss: {lh[0]:.4f} -> {lh[-1]:.4f}")
        lines.append("")
        lines.append("| epoch | train loss | val loss | val EER |")
        lines.append("|---|---|---|---|")
        for rec in summary.get("history", []):
            ve = rec.get("val_eer")
            vl = rec.get("val_loss")
            lines.append(
                f"| {rec['epoch']} | {rec['train_loss']:.4f} | "
                + (f"{vl:.4f}" if vl is not None else "-")
                + " | "
                + (f"{ve * 100:.2f}%" if ve is not None else "-")
                + " |"
            )

    failures = summary.get("failed_clips", 0)
    lines += [
        "",
        "## Data integrity",
        "",
        f"- Clips lost to decode/extract failure: {failures}",
    ]
    if failures:
        lines.append(
            "- See `feature_cache/failed_clips.json`. Cached + failed == expected "
            "for every split (asserted before training)."
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Per-language EER rests on uneven support: Gujarati has 156 test clips "
        "vs 200 for the majority languages (Tamil 191, Odia 182, Bengali 180). "
        "Treat its interval as materially wider.",
        "- Features cached in float16 (halves cache size; the Step 5 smoke test "
        "ran fp32).",
        "- 1 seed, not 3 — compressed 15-day timeline.",
        "- Metadata carries no speaker field, so speaker-disjointness across "
        "splits cannot be verified from this repo.",
        "",
        f"- Checkpoints: `{os.path.basename(checkpoint_dir)}/best_model.pt`, "
        f"`{os.path.basename(checkpoint_dir)}/last_model.pt`",
        "",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


def run_step6_resilient(
    drive_workdir: str = DEFAULT_DRIVE_WORKDIR,
    repo_dir: str = DEFAULT_REPO_DIR,
    config: Optional[TrainConfig] = None,
    require_drive: bool = True,
    status_only: bool = False,
    **step6_kwargs,
) -> Dict[str, Any]:
    """
    Single idempotent entry point for Step 6.

    Safe to re-run after any disconnect, at any point. It reports where things
    stand, then skips what is done, resumes what is partial, and continues.

    Args:
        drive_workdir: where the cache and checkpoints live. Must be on Drive to
            survive a disconnect.
        require_drive: refuse to run if Drive cannot be mounted, rather than
            silently writing to disposable local disk.
        status_only: print the status report and stop.
        **step6_kwargs: forwarded to run_step6_training (e.g.
            extract_batch_size, max_clips_per_split, store_dtype).
    """
    config = config or TrainConfig()

    print("\n" + "#" * 64)
    print("# STEP 6 — RESILIENT RUNNER")
    print("#" * 64 + "\n")

    # --- 1. Drive -----------------------------------------------------------
    mounted = ensure_drive_mounted()
    if not mounted and require_drive and in_colab():
        raise RuntimeError(
            "Google Drive is not mounted. Authorize the popup and re-run this "
            "cell. Pass require_drive=False to run against local disk anyway — "
            "but everything will be lost on the next disconnect."
        )

    workdir = drive_workdir if mounted else os.path.join("/content", "step6_local")
    cache_dir = os.path.join(workdir, "feature_cache")
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Working directory: {workdir}"
          + ("" if mounted else "   <-- LOCAL DISK, WIPED ON DISCONNECT"))

    # --- 2. Repo ------------------------------------------------------------
    if in_colab():
        try:
            repo_dir = ensure_repo(repo_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Repo sync skipped ({type(exc).__name__}: {exc})")
    else:
        repo_dir = REPO_ROOT

    # --- 3. Status ----------------------------------------------------------
    status = print_status_report(cache_dir, checkpoint_dir, config)
    if status_only:
        return {"status": status, "workdir": workdir}

    # --- 4. Do the work -----------------------------------------------------
    summary = run_step6_training(
        config=config,
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        **step6_kwargs,
    )

    # --- 5. Git-ready summary ----------------------------------------------
    summary["failed_clips"] = sum(
        (read_manifest(n, cache_dir) or {}).get("num_failed", 0)
        for n in ("train", "validation")
    )
    results_path = write_git_ready_summary(summary, repo_dir, checkpoint_dir)

    print("\n" + "=" * 64)
    print("STEP 6 COMPLETE")
    print("=" * 64)
    print(f"Validation EER : {summary['val_eer'] * 100:.2f}%")
    print(f"ECE            : {summary['ece_before']:.4f} -> {summary['ece_after']:.4f}")
    print(f"Results file   : {results_path}")
    print("\nNext action — commit:")
    print(f"  cd {repo_dir}")
    print("  git add docs/STEP6_RESULTS.md")
    print(f'  git commit -m "Step 6: validation EER '
          f'{summary["val_eer"] * 100:.2f}%"')
    print("  git push -u origin claude/ml-repo-structure-review-fpox8u")
    print("=" * 64 + "\n")

    summary["results_path"] = results_path
    summary["workdir"] = workdir
    return summary


if __name__ == "__main__":
    raise SystemExit(
        "model_pipeline.py is a library module. "
        "Run Step 6 from Colab: "
        "`from model_pipeline import run_step6_resilient; run_step6_resilient()`"
    )

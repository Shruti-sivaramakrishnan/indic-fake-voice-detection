"""
Model pipeline for AI-generated fake voice detection across 16 Indian languages.

Architecture:
    frozen wav2vec2-XLS-R-300M frontend  ->  AASIST-inspired graph-attention head

Only the head is trained. The frontend is frozen to 0 trainable parameters and is
used purely as a feature extractor.

Provenance
----------
Everything in this file except the TRAINING and PLACEHOLDER sections is consolidated
from code verified working in Google Colab. Several pieces encode fixes for specific
crashes and must not be "simplified" back:

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

Scope note (15-day compressed timeline): 1 random seed, reduced robustness matrix,
no ONNX quantization this cycle.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# SECTION 1 — Configuration
# ============================================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

SPLIT_FILE = os.path.join(REPO_ROOT, "data_pipeline", "split_ids.json")
METADATA_FILE = os.path.join(REPO_ROOT, "model_pipeline", "split_metadata_verified.json")
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
    """Hyperparameters for the head. Frontend is frozen and has none."""

    hidden_dim: int = 256
    num_gat_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 30
    grad_clip: float = 5.0
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


# ============================================================================
# SECTION 6 — Feature cache
# ============================================================================
# A full streaming pass plus frontend extraction is the expensive step. Cache the
# extracted features to disk so training/eval reruns skip both.

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
    cached_features: List[Tuple[torch.Tensor, int]],
    batch_size: int = 16,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        CachedFeatureDataset(cached_features),
        batch_size=batch_size,
        shuffle=shuffle,
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
# NOTE: this section was reconstructed from the description of the verified Colab
# loop (BCEWithLogitsLoss, AdamW over head params only, checkpoint save/reload),
# not pasted verbatim like the rest of the file. Diff it against the Colab cells
# before committing to a full Step 6 run.

def build_optimizer(model: AASISTHead, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW over head parameters only — the frontend is frozen and excluded."""
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
    grad_clip: float = 5.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for feats, mask, labels in loader:
        feats = feats.to(DEVICE)
        mask = mask.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(feats, mask=mask)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


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
    metrics: Dict[str, float],
    config: TrainConfig,
    path: str,
) -> str:
    """Head weights only — the frontend is frozen and reloaded from HF by name."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config.__dict__,
            "frontend_model_name": FRONTEND_MODEL_NAME,
        },
        path,
    )
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
        model = build_model(TrainConfig(**ckpt["config"]))

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    ckpt["model"] = model
    return ckpt


def train_model(
    train_features: List[Tuple[torch.Tensor, int]],
    val_features: List[Tuple[torch.Tensor, int]],
    config: Optional[TrainConfig] = None,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> Tuple[AASISTHead, Dict[str, Any]]:
    """
    Train the head on cached frozen features, checkpointing each epoch and
    tracking the best validation EER.
    """
    config = config or TrainConfig()
    set_seed(config.seed)

    model = build_model(config)
    optimizer = build_optimizer(model, config)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = make_dataloader(train_features, config.batch_size, shuffle=True)
    val_loader = make_dataloader(val_features, config.batch_size, shuffle=False)

    history: List[Dict[str, float]] = []
    best_eer = float("inf")
    best_path = os.path.join(checkpoint_dir, "best_model.pt")

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.grad_clip
        )
        val_metrics = evaluate(model, val_loader, criterion)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss", float("nan")),
            "val_eer": val_metrics["eer"],
        }
        history.append(record)

        print(
            f"epoch {epoch:>3}/{config.num_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={record['val_loss']:.4f}  "
            f"val_eer={val_metrics['eer'] * 100:.2f}%"
        )

        save_checkpoint(
            model, optimizer, epoch, val_metrics, config,
            os.path.join(checkpoint_dir, "last_model.pt"),
        )

        if val_metrics["eer"] < best_eer:
            best_eer = val_metrics["eer"]
            save_checkpoint(model, optimizer, epoch, val_metrics, config, best_path)
            print(f"  new best val EER {best_eer * 100:.2f}% -> {best_path}")

    return model, {"history": history, "best_eer": best_eer, "best_path": best_path}


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
# PLACEHOLDER. This is the end-to-end script that ties Sections 3-10 together and
# produces the first real EER numbers (per TEAM_SYNC_LOG, nothing is reportable
# before this runs).
#
# Intended sequence:
#
#   1.  assert_split_integrity()               — refuse to run on a corrupted split
#   2.  verify_frontend_frozen()               — assert 0 trainable frontend params
#   3.  load_frozen_split_sets() / build_label_lookup()
#   4.  stream_and_filter_by_ids(..., need_audio=True)   — the ~20+ min pass
#   5.  build_feature_cache() + save_feature_cache() per split
#         (reuse load_feature_cache() on rerun — do not re-stream)
#   6.  train_model()                          — checkpoints each epoch
#   7.  load best checkpoint, fit TemperatureScaler on VALIDATION logits only
#   8.  final test-set pass: EER, ECE, per-language breakdown
#   9.  rerun data_pipeline/leakage_audit.py once degraded audio exists
#  10.  write results + update docs/TEAM_SYNC_LOG.md
#
# Scope this cycle: 1 seed (SEED = 42), reduced robustness matrix, no ONNX
# quantization.


def run_step6_training(config: Optional[TrainConfig] = None) -> Dict[str, Any]:
    """PLACEHOLDER — full Step 6 training run. See the sequence above."""
    raise NotImplementedError(
        "Step 6 full training run is not wired up yet."
    )


if __name__ == "__main__":
    raise SystemExit(
        "model_pipeline.py is a library module. "
        "Step 6 (run_step6_training) is not implemented yet."
    )

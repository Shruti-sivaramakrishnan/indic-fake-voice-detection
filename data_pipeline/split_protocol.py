"""Verification helpers for reportable Voice Guard split artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Set


PROTOCOL_VERSION = "v2-group-disjoint"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_audit_path(split_file: Path) -> Path:
    return split_file.with_suffix(".audit.json")


def verify_split_audit(
    split_file: Path | str,
    audit_file: Optional[Path | str] = None,
    metadata_file: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Verify that a v2 audit is complete and bound to the exact split bytes."""
    split_path = Path(split_file)
    audit_path = Path(audit_file) if audit_file else default_audit_path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not audit_path.exists():
        raise FileNotFoundError(
            f"Split audit not found: {audit_path}. Generate the split with "
            "data_pipeline/build_group_safe_split.py."
        )

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported split protocol {audit.get('protocol_version')!r}; "
            f"expected {PROTOCOL_VERSION!r}."
        )
    if audit.get("split_sha256") != sha256_file(split_path):
        raise ValueError(
            "Split audit SHA-256 does not match the split file. Regenerate the "
            "split and audit together."
        )
    if int(audit.get("group_overlap_count", -1)) != 0:
        raise ValueError("Split audit reports cross-split group overlap.")
    if not audit.get("all_splits_have_every_language_and_label", False):
        raise ValueError(
            "Split audit does not contain every language/label stratum in every split."
        )
    if int(audit.get("total_clips", -1)) <= 0:
        raise ValueError("Split audit has an invalid total clip count.")
    if not str(audit.get("group_strategy", "")).strip():
        raise ValueError("Split audit is missing its grouping strategy.")
    if not str(audit.get("source_sha256", "")).strip():
        raise ValueError("Split audit is missing source-metadata provenance.")
    if metadata_file is not None:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Split metadata not found: {metadata_path}")
        if audit["source_sha256"] != sha256_file(metadata_path):
            raise ValueError(
                "Split audit source SHA-256 does not match the metadata file. "
                "Use the metadata artifact that produced this split."
            )
    return audit


def select_stratified_ids(
    clip_ids: Iterable[str],
    labels: Dict[str, int],
    languages: Dict[str, str],
    limit: int,
) -> Set[str]:
    """Select a deterministic round-robin rehearsal subset across all strata."""
    if limit <= 0:
        raise ValueError("Rehearsal limit must be positive.")

    buckets: Dict[tuple[str, int], list[str]] = defaultdict(list)
    for clip_id in sorted(set(clip_ids)):
        if clip_id not in labels or clip_id not in languages:
            raise KeyError(f"Clip {clip_id!r} is absent from verified metadata.")
        buckets[(languages[clip_id], int(labels[clip_id]))].append(clip_id)

    selected: Set[str] = set()
    positions = {stratum: 0 for stratum in buckets}
    strata = sorted(buckets)
    target = min(limit, sum(len(values) for values in buckets.values()))
    while len(selected) < target:
        progressed = False
        for stratum in strata:
            position = positions[stratum]
            if position >= len(buckets[stratum]):
                continue
            selected.add(buckets[stratum][position])
            positions[stratum] += 1
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            break
    return selected

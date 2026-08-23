"""Build a deterministic group-disjoint v2 split from enriched metadata.

Do not generate a v2 split from clip IDs alone unless the proxy limitation is
explicitly accepted. Preferred metadata includes a real speaker/source/session
or prompt identifier. The output keeps every group in exactly one split and
writes a companion audit JSON describing the provenance and balance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SPLIT_NAMES = ("train", "validation", "test")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def id_family(clip_id: str) -> str:
    return re.sub(r"[_-]?\d+$", "", clip_id.strip()).casefold()


def _records_from_json(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("JSON list rows must be objects.")
            yield row
        return
    if isinstance(payload, dict) and set(payload) == set(SPLIT_NAMES):
        for rows in payload.values():
            for row in rows:
                if isinstance(row, dict):
                    yield row
                elif isinstance(row, list) and len(row) >= 3:
                    yield {"id": row[0], "language": row[1], "label": row[2]}
                else:
                    raise ValueError("Unsupported split-metadata row format.")
        return
    raise ValueError("JSON must be a list of objects or a train/validation/test mapping.")


def read_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if suffix == ".json":
        return list(_records_from_json(json.loads(path.read_text(encoding="utf-8"))))
    raise ValueError("Metadata must be CSV, JSONL/NDJSON, or JSON.")


def canonical_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().casefold()
    if text in {"0", "real", "bonafide", "bona_fide", "genuine"}:
        return 0
    if text in {"1", "fake", "spoof", "synthetic", "deepfake"}:
        return 1
    raise ValueError(f"Unsupported label {value!r}; expected real/0 or fake/1.")


def prepare_rows(
    rows: Sequence[Dict[str, Any]],
    group_columns: Sequence[str],
    allow_id_family_proxy: bool,
) -> Tuple[List[Dict[str, Any]], str]:
    prepared = []
    seen_ids = set()
    strategy = "columns:" + "+".join(group_columns) if group_columns else "id_family_proxy"
    for index, row in enumerate(rows, start=1):
        clip_id = str(row.get("id", row.get("clip_id", ""))).strip()
        language = str(row.get("language", "")).strip()
        if not clip_id or not language or "label" not in row:
            raise ValueError(f"Row {index} needs id/clip_id, language, and label.")
        if clip_id in seen_ids:
            raise ValueError(f"Duplicate clip ID {clip_id!r} in metadata.")
        seen_ids.add(clip_id)

        if group_columns:
            missing = [column for column in group_columns if not str(row.get(column, "")).strip()]
            if missing:
                raise ValueError(f"Row {index} is missing group columns: {missing}")
            group_id = "||".join(normalize_text(row[column]) for column in group_columns)
        elif allow_id_family_proxy:
            group_id = id_family(clip_id)
        else:
            raise ValueError(
                "No trustworthy group columns were supplied. Pass --group-columns "
                "speaker_id/source_id/prompt_id fields, or explicitly use "
                "--allow-id-family-proxy for an acknowledged candidate split."
            )

        prepared.append(
            {
                "id": clip_id,
                "language": language,
                "label": canonical_label(row["label"]),
                "group_id": group_id,
            }
        )
    return prepared, strategy


def assign_groups(
    rows: Sequence[Dict[str, Any]],
    ratios: Dict[str, float],
    seed: int,
) -> Dict[str, List[str]]:
    if set(ratios) != set(SPLIT_NAMES) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must contain train/validation/test and sum to 1.")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    totals: Counter = Counter()
    for row in rows:
        grouped[row["group_id"]].append(row)
        totals[(row["language"], row["label"])] += 1

    rng = random.Random(seed)
    tie_breakers = {group_id: rng.random() for group_id in grouped}
    ordered_groups = sorted(
        grouped,
        key=lambda key: (-len(grouped[key]), tie_breakers[key], key),
    )
    targets = {
        split: {stratum: total * ratios[split] for stratum, total in totals.items()}
        for split in SPLIT_NAMES
    }
    counts = {split: Counter() for split in SPLIT_NAMES}
    sizes = Counter()
    assignments: Dict[str, str] = {}

    def score(candidate: str, group_counts: Counter, group_size: int) -> float:
        error = 0.0
        for split in SPLIT_NAMES:
            for stratum, total in totals.items():
                value = counts[split][stratum]
                if split == candidate:
                    value += group_counts[stratum]
                target = targets[split][stratum]
                error += ((value - target) / max(target, 1.0)) ** 2
            size = sizes[split] + (group_size if split == candidate else 0)
            size_target = len(rows) * ratios[split]
            error += 0.2 * ((size - size_target) / max(size_target, 1.0)) ** 2
        return error

    for group_id in ordered_groups:
        group_rows = grouped[group_id]
        group_counts = Counter((row["language"], row["label"]) for row in group_rows)
        best_split = min(
            SPLIT_NAMES,
            key=lambda split: (score(split, group_counts, len(group_rows)), SPLIT_NAMES.index(split)),
        )
        assignments[group_id] = best_split
        counts[best_split].update(group_counts)
        sizes[best_split] += len(group_rows)

    output = {split: [] for split in SPLIT_NAMES}
    for row in rows:
        output[assignments[row["group_id"]]].append(row["id"])
    for split in SPLIT_NAMES:
        output[split].sort()
        if not output[split]:
            raise RuntimeError(f"Greedy assignment produced an empty {split} split.")
    return output


def audit_output(
    rows: Sequence[Dict[str, Any]],
    splits: Dict[str, List[str]],
) -> Dict[str, Any]:
    by_id = {row["id"]: row for row in rows}
    all_strata = {
        f"{row['language']}|{row['label']}"
        for row in rows
    }
    group_owner: Dict[str, str] = {}
    overlap = []
    distribution: Dict[str, Any] = {}
    for split, clip_ids in splits.items():
        strata = Counter()
        for clip_id in clip_ids:
            row = by_id[clip_id]
            group_id = row["group_id"]
            previous = group_owner.setdefault(group_id, split)
            if previous != split:
                overlap.append({"group_id": group_id, "splits": sorted({previous, split})})
            strata[f"{row['language']}|{row['label']}"] += 1
        distribution[split] = {
            "clips": len(clip_ids),
            "groups": len({by_id[clip_id]["group_id"] for clip_id in clip_ids}),
            "strata": dict(sorted(strata.items())),
            "missing_strata": sorted(all_strata - set(strata)),
        }
    return {
        "total_clips": len(rows),
        "total_groups": len({row["group_id"] for row in rows}),
        "group_overlap_count": len(overlap),
        "group_overlap_examples": overlap[:20],
        "all_splits_have_every_language_and_label": all(
            not entry["missing_strata"] for entry in distribution.values()
        ),
        "distribution": distribution,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data_pipeline/split_ids_v2.json"))
    parser.add_argument("--group-columns", nargs="+", default=[])
    parser.add_argument("--allow-id-family-proxy", action="store_true")
    parser.add_argument(
        "--allow-missing-strata",
        action="store_true",
        help="Write an exploratory split even if a language/label is absent from a split.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_bytes = args.metadata.read_bytes()
    rows, strategy = prepare_rows(
        read_records(args.metadata),
        args.group_columns,
        args.allow_id_family_proxy,
    )
    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    splits = assign_groups(rows, ratios, args.seed)
    audit = audit_output(rows, splits)
    if audit["group_overlap_count"]:
        raise RuntimeError("Internal error: generated split has cross-split group overlap.")
    if not audit["all_splits_have_every_language_and_label"] and not args.allow_missing_strata:
        missing = {
            split: entry["missing_strata"]
            for split, entry in audit["distribution"].items()
            if entry["missing_strata"]
        }
        raise ValueError(
            "The available groups cannot preserve every language/label in every "
            f"split: {missing}. Supply finer trustworthy group metadata. Use "
            "--allow-missing-strata only to inspect a non-reportable candidate."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    audit_path = args.output.with_suffix(".audit.json")
    audit_payload = {
        "protocol_version": "v2-group-disjoint",
        "group_strategy": strategy,
        "source_metadata": str(args.metadata.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed": args.seed,
        "ratios": ratios,
        **audit,
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {audit_path}")
    print(json.dumps(audit["distribution"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

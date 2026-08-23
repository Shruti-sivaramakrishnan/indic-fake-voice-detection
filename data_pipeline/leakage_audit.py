"""Audit exact IDs, source-family proxies, and degraded-audio lineage.

The original audit checked only exact clip IDs. That is necessary but not
sufficient: clips with the same source/emotion/speaker-like ID prefix are spread
across all three v1 splits. This audit makes that limitation visible and fails by
default until a group-disjoint v2 split is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from data_pipeline.split_protocol import verify_split_audit
except ModuleNotFoundError:  # direct `python data_pipeline/leakage_audit.py`
    from split_protocol import verify_split_audit


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_FILE = REPO_ROOT / "data_pipeline" / "split_ids.json"
EXPECTED_TOTAL = 31_102
SPLIT_NAMES = ("train", "validation", "test")


def load_splits(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen split not found: {path}")
    splits = json.loads(path.read_text(encoding="utf-8"))
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"Expected split keys {set(SPLIT_NAMES)}, found {set(splits)}")
    return {name: [str(value) for value in splits[name]] for name in SPLIT_NAMES}


def id_family(clip_id: str) -> str:
    """Conservative source-family proxy: remove only the trailing clip number."""
    return re.sub(r"[_-]?\d+$", "", clip_id.strip()).casefold()


def check_exact_ids(splits: Dict[str, List[str]], expected_total: int) -> bool:
    passed = True
    sets = {name: set(values) for name, values in splits.items()}
    print("\n=== EXACT ID INTEGRITY ===")
    for name, values in splits.items():
        duplicates = [key for key, count in Counter(values).items() if count > 1]
        print(f"{name:<10}: {len(values):>6,} rows, {len(duplicates):>3} duplicates")
        passed = passed and not duplicates

    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = sets[left] & sets[right]
            print(f"{left} intersect {right}: {len(overlap):,} exact IDs")
            passed = passed and not overlap

    unique = set().union(*sets.values())
    print(f"Unique IDs: {len(unique):,} (expected {expected_total:,})")
    return passed and len(unique) == expected_total


def check_group_overlap(splits: Dict[str, List[str]]) -> bool:
    print("\n=== SOURCE-FAMILY PROXY OVERLAP ===")
    families = {
        name: {id_family(clip_id) for clip_id in values}
        for name, values in splits.items()
    }
    passed = True
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            shared = families[left] & families[right]
            left_clips = sum(id_family(value) in shared for value in splits[left])
            right_clips = sum(id_family(value) in shared for value in splits[right])
            print(
                f"{left} vs {right}: {len(shared):,} shared families; "
                f"{left_clips:,}/{len(splits[left]):,} {left} clips and "
                f"{right_clips:,}/{len(splits[right]):,} {right} clips belong "
                "to shared families"
            )
            if shared:
                print(f"  examples: {', '.join(sorted(shared)[:5])}")
                passed = False
    return passed


def _manifest_paths() -> Iterable[Path]:
    degraded = REPO_ROOT / "data_pipeline" / "degraded"
    return sorted(degraded.glob("*.jsonl")) if degraded.exists() else []


def check_degraded_lineage(splits: Dict[str, List[str]]) -> bool:
    print("\n=== DEGRADED-AUDIO LINEAGE ===")
    owner = {
        clip_id: split_name
        for split_name, clip_ids in splits.items()
        for clip_id in clip_ids
    }
    paths = list(_manifest_paths())
    if not paths:
        print("No degraded manifests found.")
        return True

    passed = True
    seen = set()
    for path in paths:
        rows = 0
        errors = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {line_number}: invalid JSON ({exc})")
                    continue
                clip_id = str(row.get("id", ""))
                declared_split = str(row.get("split", ""))
                degradation = str(row.get("degradation", ""))
                key = (clip_id, degradation)
                if clip_id not in owner:
                    errors.append(f"line {line_number}: unknown id {clip_id!r}")
                elif declared_split != owner[clip_id]:
                    errors.append(
                        f"line {line_number}: {clip_id} declared {declared_split}, "
                        f"belongs to {owner[clip_id]}"
                    )
                if key in seen:
                    errors.append(f"line {line_number}: duplicate lineage {key}")
                seen.add(key)
        print(f"{path.name}: {rows:,} rows, {len(errors):,} lineage errors")
        for error in errors[:5]:
            print(f"  {error}")
        passed = passed and not errors
    return passed


def run_audit(
    split_file: Path = DEFAULT_SPLIT_FILE,
    expected_total: int = EXPECTED_TOTAL,
    allow_group_overlap: bool = False,
) -> bool:
    print("=" * 72)
    print("VOICE GUARD SPLIT AND LINEAGE AUDIT")
    print("=" * 72)
    print(f"Split: {split_file}")
    splits = load_splits(split_file)
    exact_ok = check_exact_ids(splits, expected_total)
    proxy_groups_ok = check_group_overlap(splits)
    if split_file.resolve() == DEFAULT_SPLIT_FILE.resolve():
        groups_ok = proxy_groups_ok
        lineage_ok = check_degraded_lineage(splits)
    else:
        print("\n=== DECLARED V2 GROUP PROTOCOL ===")
        try:
            audit = verify_split_audit(split_file)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"FAIL: {exc}")
            groups_ok = False
        else:
            print(f"protocol: {audit['protocol_version']}")
            print(f"strategy: {audit['group_strategy']}")
            print(f"groups: {audit['total_groups']:,}; declared overlap: 0")
            if not proxy_groups_ok:
                print(
                    "NOTE: filename-family proxies still overlap. They are a "
                    "diagnostic, not the declared grouping source for this split."
                )
            groups_ok = True
        print("\n=== DEGRADED-AUDIO LINEAGE ===")
        print(
            "Skipped legacy v1 degraded manifests for this alternate split. "
            "Regenerate augmentations from the new split before training."
        )
        lineage_ok = True

    historical_exception = False
    if (
        split_file.resolve() == DEFAULT_SPLIT_FILE.resolve()
        and not groups_ok
        and allow_group_overlap
    ):
        print("\nWARNING: group overlap was explicitly allowed for historical v1 review.")
        groups_ok = True
        historical_exception = True

    passed = exact_ok and groups_ok and lineage_ok
    print("\n" + "=" * 72)
    if passed and historical_exception:
        print("PASS WITH EXCEPTION: exact IDs/lineage pass; known v1 group overlap remains.")
    elif passed:
        print("PASS: split is exact-ID and group-disjoint.")
    else:
        print("FAIL: split is not safe for a new reportable training run.")
    print("=" * 72)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--expected-total", type=int, default=EXPECTED_TOTAL)
    parser.add_argument(
        "--allow-known-v1-group-overlap",
        action="store_true",
        help="Allow the known proxy-family overlap only for historical v1 review.",
    )
    args = parser.parse_args()
    raise SystemExit(
        0
        if run_audit(
            args.split_file,
            args.expected_total,
            allow_group_overlap=args.allow_known_v1_group_overlap,
        )
        else 1
    )

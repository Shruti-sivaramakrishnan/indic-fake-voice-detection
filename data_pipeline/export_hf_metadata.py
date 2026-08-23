"""Export label/language/prompt metadata without downloading or decoding audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_DATASET = "SherryT997/IndicTTS-Deepfake-Challenge-Data"
DEFAULT_SPLIT = "train"
EXPECTED_TOTAL = 31_102


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def prompt_id(language: str, text: str) -> str:
    normalized = f"{normalize_text(language)}\0{normalize_text(text)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def id_family(clip_id: str) -> str:
    return re.sub(r"[_-]?\d+$", "", clip_id.strip()).casefold()


def metadata_record(example: Dict[str, Any]) -> Dict[str, Any]:
    clip_id = str(example.get("id", "")).strip()
    language = str(example.get("language", "")).strip()
    text = str(example.get("text", "")).strip()
    label = int(example.get("is_tts", -1))
    if not clip_id or not language or not text:
        raise ValueError("Every row needs non-empty id, language, and text fields.")
    if label not in {0, 1}:
        raise ValueError(f"Clip {clip_id!r} has unsupported is_tts label {label}.")
    return {
        "id": clip_id,
        "language": language,
        "label": label,
        "prompt_id": prompt_id(language, text),
        "source_family_proxy": id_family(clip_id),
        "text_sha256": hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest(),
    }


def iter_hf_metadata(dataset_name: str, split: str) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    if "audio" in dataset.column_names:
        dataset = dataset.remove_columns("audio")
    for example in dataset:
        yield metadata_record(example)


def export_metadata(
    output: Path,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    expected_total: Optional[int] = EXPECTED_TOTAL,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    seen_ids = set()
    strata: Counter[str] = Counter()
    prompt_ids = set()
    count = 0

    digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        for record in iter_hf_metadata(dataset_name, split):
            if record["id"] in seen_ids:
                raise ValueError(f"Duplicate clip ID {record['id']!r} in upstream metadata.")
            seen_ids.add(record["id"])
            prompt_ids.add(record["prompt_id"])
            strata[f"{record['language']}|{record['label']}"] += 1
            line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            stream.write(line)
            digest.update(line)
            count += 1
            if max_rows is not None and count >= max_rows:
                break

    if max_rows is None and expected_total is not None and count != expected_total:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Exported {count:,} rows; expected {expected_total:,}. Refusing a partial artifact."
        )
    temporary.replace(output)

    summary = {
        "dataset": dataset_name,
        "split": split,
        "rows": count,
        "unique_prompt_groups": len(prompt_ids),
        "strata": dict(sorted(strata.items())),
        "metadata_sha256": digest.hexdigest(),
        "audio_downloaded": False,
        "grouping_scope": "prompt-disjoint; speaker/source/generator lineage unavailable upstream",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--expected-total", type=int, default=EXPECTED_TOTAL)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    summary = export_metadata(
        output=args.output,
        dataset_name=args.dataset,
        split=args.split,
        expected_total=args.expected_total,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

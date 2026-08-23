import hashlib
import json

import pytest

from data_pipeline.build_group_safe_split import audit_output, assign_groups, prepare_rows
from data_pipeline.export_hf_metadata import metadata_record, normalize_text, prompt_id
from data_pipeline.split_protocol import select_stratified_ids, verify_split_audit


def test_prompt_id_is_stable_across_unicode_and_whitespace():
    first = prompt_id("Tamil", "  ஒரு   சோதனை ")
    second = prompt_id(" tamil ", "ஒரு சோதனை")
    assert first == second
    assert normalize_text("  A  B ") == "a b"


def test_metadata_record_rejects_hidden_test_label():
    with pytest.raises(ValueError, match="unsupported"):
        metadata_record(
            {"id": "clip-1", "language": "Hindi", "text": "नमस्ते", "is_tts": -1}
        )


def test_multiple_group_columns_form_connected_components():
    rows = [
        {"id": "a", "language": "Hindi", "label": 0, "speaker": "s1", "prompt": "p1"},
        {"id": "b", "language": "Hindi", "label": 1, "speaker": "s1", "prompt": "p2"},
        {"id": "c", "language": "Hindi", "label": 0, "speaker": "s2", "prompt": "p2"},
        {"id": "d", "language": "Hindi", "label": 1, "speaker": "s3", "prompt": "p3"},
    ]
    prepared, strategy = prepare_rows(rows, ["speaker", "prompt"], False)
    groups = {row["id"]: row["group_id"] for row in prepared}
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]
    assert strategy.startswith("connected_components:any_shared")


def test_split_audit_is_bound_to_exact_split_bytes(tmp_path):
    rows = [
        {"id": f"h0-{i}", "language": "Hindi", "label": 0, "prompt": f"r{i}"}
        for i in range(20)
    ] + [
        {"id": f"h1-{i}", "language": "Hindi", "label": 1, "prompt": f"f{i}"}
        for i in range(20)
    ]
    prepared, strategy = prepare_rows(rows, ["prompt"], False)
    splits = assign_groups(
        prepared,
        {"train": 0.6, "validation": 0.2, "test": 0.2},
        seed=42,
    )
    audit = audit_output(prepared, splits)
    split_path = tmp_path / "split.json"
    split_bytes = (json.dumps(splits, indent=2) + "\n").encode()
    split_path.write_bytes(split_bytes)
    audit.update(
        {
            "protocol_version": "v2-group-disjoint",
            "group_strategy": strategy,
            "source_sha256": "abc123",
            "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        }
    )
    split_path.with_suffix(".audit.json").write_text(json.dumps(audit))
    assert verify_split_audit(split_path)["group_overlap_count"] == 0

    split_path.write_text(split_path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_split_audit(split_path)


def test_split_audit_is_bound_to_source_metadata(tmp_path):
    split_path = tmp_path / "split.json"
    metadata_path = tmp_path / "metadata.jsonl"
    split_path.write_text('{"train": ["a"], "validation": ["b"], "test": ["c"]}\n')
    metadata_path.write_text('{"id": "a"}\n')
    audit = {
        "protocol_version": "v2-group-disjoint",
        "group_strategy": "connected_components:any_shared:prompt_id",
        "source_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "group_overlap_count": 0,
        "all_splits_have_every_language_and_label": True,
        "total_clips": 3,
    }
    split_path.with_suffix(".audit.json").write_text(json.dumps(audit))
    verify_split_audit(split_path, metadata_file=metadata_path)
    metadata_path.write_text('{"id": "different"}\n')
    with pytest.raises(ValueError, match="source SHA-256"):
        verify_split_audit(split_path, metadata_file=metadata_path)


def test_rehearsal_selection_round_robins_language_and_label():
    ids = [f"h-{label}-{index}" for label in (0, 1) for index in range(5)]
    labels = {clip_id: int(clip_id.split("-")[1]) for clip_id in ids}
    languages = {clip_id: "Hindi" for clip_id in ids}
    selected = select_stratified_ids(ids, labels, languages, limit=4)
    assert len(selected) == 4
    assert sum(labels[clip_id] == 0 for clip_id in selected) == 2
    assert sum(labels[clip_id] == 1 for clip_id in selected) == 2

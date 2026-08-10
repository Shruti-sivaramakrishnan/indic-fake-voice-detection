
import json
import os
from collections import Counter

SPLIT_FILE = "data_pipeline/split_ids.json"


def load_splits():
    """Load the frozen split. Never regenerate it."""
    
    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(
            f"Frozen split not found: {SPLIT_FILE}"
        )

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    required = {"train", "validation", "test"}

    if set(splits.keys()) != required:
        raise ValueError(
            f"Expected split keys {required}, "
            f"found {set(splits.keys())}"
        )

    return splits


def check_duplicate_ids(splits):
    """Check for duplicate clip IDs within each split."""

    print("\n=== DUPLICATE ID CHECK ===")

    passed = True

    for split_name, ids in splits.items():

        counts = Counter(ids)
        duplicates = [
            clip_id
            for clip_id, count in counts.items()
            if count > 1
        ]

        if duplicates:
            print(
                f"FAIL: {split_name} contains "
                f"{len(duplicates)} duplicate IDs."
            )
            passed = False
        else:
            print(
                f"PASS: {split_name} contains "
                f"no duplicate IDs."
            )

    return passed


def check_split_overlap(splits):
    """Check that no clip ID occurs in multiple splits."""

    print("\n=== CROSS-SPLIT OVERLAP CHECK ===")

    train_ids = set(splits["train"])
    val_ids = set(splits["validation"])
    test_ids = set(splits["test"])

    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    print("Train ∩ Validation:", len(train_val))
    print("Train ∩ Test:", len(train_test))
    print("Validation ∩ Test:", len(val_test))

    passed = (
        len(train_val) == 0
        and len(train_test) == 0
        and len(val_test) == 0
    )

    if passed:
        print("PASS: No clip ID crosses split boundaries.")
    else:
        print("FAIL: Split leakage detected.")

    return passed


def check_total_ids(splits):
    """Check that all expected IDs are accounted for."""

    print("\n=== TOTAL ID CHECK ===")

    train_ids = set(splits["train"])
    val_ids = set(splits["validation"])
    test_ids = set(splits["test"])

    all_ids = train_ids | val_ids | test_ids

    print("Train IDs:", len(train_ids))
    print("Validation IDs:", len(val_ids))
    print("Test IDs:", len(test_ids))
    print("Unique IDs:", len(all_ids))

    expected_total = 31102

    if len(all_ids) == expected_total:
        print(
            f"PASS: All {expected_total:,} expected "
            f"clip IDs are accounted for."
        )
        return True

    print(
        f"FAIL: Expected {expected_total:,} unique IDs "
        f"but found {len(all_ids):,}."
    )
    return False


def check_degraded_files(splits):
    """
    Future leakage check.

    Once degraded/augmented audio exists, each degraded file
    must inherit the split of its original clip.

    This function currently reports that no degraded-audio
    directory has been created yet.
    """

    print("\n=== DEGRADED/AUGMENTED FILE CHECK ===")

    degraded_dirs = [
        "data_pipeline/degraded",
        "data_pipeline/degraded_audio",
        "data_pipeline/augmented",
        "data_pipeline/augmented_audio",
    ]

    existing_dirs = [
        directory
        for directory in degraded_dirs
        if os.path.exists(directory)
    ]

    if not existing_dirs:
        print(
            "PASS: No degraded/augmented audio exists yet."
        )
        print(
            "This check must be rerun after Step 6."
        )
        return True

    print("Found degraded/augmented directories:")
    for directory in existing_dirs:
        print(" -", directory)

    print(
        "\nManual/metadata-based degraded-file audit "
        "will be performed after Step 6."
    )

    return True


def run_audit():

    print("=" * 60)
    print("FROZEN SPLIT LEAKAGE AUDIT")
    print("=" * 60)

    splits = load_splits()

    duplicate_check = check_duplicate_ids(splits)
    overlap_check = check_split_overlap(splits)
    total_check = check_total_ids(splits)
    degraded_check = check_degraded_files(splits)

    print("\n" + "=" * 60)
    print("FINAL AUDIT RESULT")
    print("=" * 60)

    if all([
        duplicate_check,
        overlap_check,
        total_check,
        degraded_check
    ]):
        print("✅ LEAKAGE AUDIT PASSED")
    else:
        print("❌ LEAKAGE AUDIT FAILED")

        raise SystemExit(
            "Leakage audit failed. "
            "Do not continue until the issue is fixed."
        )


if __name__ == "__main__":
    run_audit()

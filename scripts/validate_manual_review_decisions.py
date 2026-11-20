"""Read-only check: python -B scripts/validate_manual_review_decisions.py."""

import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ("clothes", "11310"),
    ("computer", "12311"),
    ("drink", "17710"),
    ("drink", "17711"),
    ("go", "24941"),
}
REQUIRED_COLUMNS = {"gloss", "video_id", "decision", "review_status", "notes",
                    "sign_boundary_verification"}


def validate_decisions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("Required decision columns are missing.")
        rows = list(reader)
    if len(rows) != 5:
        raise ValueError(f"Expected exactly five decisions; found {len(rows)}.")
    video_ids = [row["video_id"] for row in rows]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("Duplicate video IDs found.")
    if {(row["gloss"], row["video_id"]) for row in rows} != EXPECTED:
        raise ValueError("Clip identifiers do not match the five requested clips.")
    for row in rows:
        identifier = f"{row['gloss']}/{row['video_id']}"
        for field, expected in (
            ("decision", "provisional_keep"),
            ("review_status", "provisional_sampled_frame_review"),
            ("sign_boundary_verification", "pending"),
        ):
            if row[field] != expected:
                raise ValueError(f"{identifier}: {field} must be {expected!r}.")
        if not row["notes"] or not row["notes"].strip():
            raise ValueError(f"{identifier}: observation notes are missing.")


def main():
    path = PROJECT_ROOT / "dataset/metadata/manual_review_decisions.csv"
    try:
        validate_decisions(path)
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: exactly five provisional_keep decisions; no duplicate video IDs.")
    print("PASS: all review statuses are provisional, notes are present, and sign-boundary verification is pending.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Explore WLASL metadata without opening videos or contacting any URLs.

From the project root, with the project's existing Python environment active:
    python scripts/explore_dataset.py

Requires pandas in that environment. Outputs overwrite dataset/metadata/
wlasl_instances.csv and wlasl_summary.csv. Paths work from any working directory.
"""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_COLUMNS = [
    "gloss", "video_id", "signer_id", "split", "url", "frame_start", "frame_end"
]


def find_metadata(dataset_dir):
    """Prefer start_kit; choose deterministically and report the chosen file."""
    matches = sorted(
        dataset_dir.rglob("WLASL_v*.json"),
        key=lambda path: ("start_kit" not in path.relative_to(dataset_dir).parts,
                          path.as_posix()),
    )
    if not matches:
        raise FileNotFoundError(
            f"No WLASL_v*.json found under {dataset_dir}. "
            "Place the WLASL metadata JSON there, preferably in start_kit."
        )
    if len(matches) > 1:
        print(f"Found {len(matches)} metadata files; using only {matches[0]}.")
    return matches[0]


def flatten_entries(entries):
    """Return rows and gloss names; skip malformed records with their locations."""
    if not isinstance(entries, list):
        raise ValueError("WLASL JSON must be a list of gloss entries.")

    rows = []
    glosses = set()
    issues = 0

    def report(location, message):
        nonlocal issues
        issues += 1
        print(f"WARNING: {location}: {message}", file=sys.stderr)

    for entry_index, entry in enumerate(entries):
        location = f"entry[{entry_index}]"
        if not isinstance(entry, dict):
            report(location, "expected an object; skipped.")
            continue
        gloss = entry.get("gloss")
        if not isinstance(gloss, str) or not gloss.strip():
            report(location, "missing or invalid gloss; skipped.")
            continue
        glosses.add(gloss)
        instances = entry.get("instances")
        if not isinstance(instances, list):
            report(location, "missing or invalid instances list; skipped instances.")
            continue

        # Prefix entry metadata so it cannot overwrite instance fields.
        entry_metadata = {
            f"entry_{key}": value for key, value in entry.items()
            if key not in ("gloss", "instances")
        }
        for instance_index, instance in enumerate(instances):
            instance_location = f"{location}.instances[{instance_index}]"
            if not isinstance(instance, dict):
                report(instance_location, "expected an object; skipped.")
                continue
            row = {**entry_metadata, **instance, "gloss": gloss}
            for column in INSTANCE_COLUMNS:
                row.setdefault(column, None)
            for column in INSTANCE_COLUMNS[1:]:
                value = row[column]
                if value is not None and (
                    isinstance(value, (dict, list, bool))
                    or (column in ("split", "url") and not isinstance(value, str))
                ):
                    report(instance_location, f"invalid {column}; treated as missing.")
                    row[column] = None
            if isinstance(row["signer_id"], str) and not row["signer_id"].strip():
                row["signer_id"] = None
            # Keep leading zeros in video IDs and unify numeric/string signer IDs.
            for column in ("video_id", "signer_id"):
                if row[column] is not None:
                    row[column] = str(row[column])
            rows.append(row)

    print(f"Metadata issues reported: {issues}")
    return rows, sorted(glosses)


def summarize_instances(instances, glosses):
    """Include zero-instance glosses and exclude missing IDs from signer counts."""
    import pandas as pd

    grouped = instances.groupby("gloss")
    summary = pd.DataFrame(index=pd.Index(glosses, name="gloss"))
    summary["instance_count"] = grouped.size()
    summary["signer_count"] = grouped["signer_id"].nunique()
    for split in ("train", "val", "test"):
        summary[f"{split}_count"] = (
            instances.loc[instances["split"].eq(split)].groupby("gloss").size()
        )
    summary = summary.fillna(0).astype(int).reset_index()
    summary["other_or_missing_split_count"] = (
        summary["instance_count"]
        - summary[["train_count", "val_count", "test_count"]].sum(axis=1)
    )
    return summary.sort_values(
        ["instance_count", "gloss"], ascending=[False, True]
    ).reset_index(drop=True)


def main():
    try:
        import pandas as pd
    except ImportError:
        print(
            "ERROR: pandas could not be imported. Run this script with the existing "
            "project Python environment that contains pandas. "
            "No dependencies have been installed or changed.", file=sys.stderr,
        )
        return 1

    try:
        metadata_path = find_metadata(PROJECT_ROOT / "dataset" / "WLASL")
        print(f"Metadata source: {metadata_path}")
        with metadata_path.open(encoding="utf-8-sig") as metadata_file:
            entries = json.load(metadata_file)
        rows, glosses = flatten_entries(entries)
        if not glosses:
            raise ValueError("No valid gloss entries found; CSV files were not written.")
        instances = pd.DataFrame(rows).reindex(columns=INSTANCE_COLUMNS + sorted(
            {key for row in rows for key in row} - set(INSTANCE_COLUMNS)
        ))
        summary = summarize_instances(instances, glosses)
        output_dir = PROJECT_ROOT / "dataset" / "metadata"
        output_dir.mkdir(parents=True, exist_ok=True)
        instances.to_csv(output_dir / "wlasl_instances.csv", index=False)
        summary.to_csv(output_dir / "wlasl_summary.csv", index=False)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"\nTotal glosses: {len(glosses):,}")
    print(f"Total metadata instances: {len(instances):,}")
    print(f"Total distinct known signers: {instances['signer_id'].nunique():,}")
    print(f"Instances with missing signer IDs: {instances['signer_id'].isna().sum():,}")
    print("\nTop 30 glosses by metadata instance count:")
    print(summary.head(30).to_string(index=False))
    candidates = summary.loc[
        (summary["instance_count"] >= 15) & (summary["signer_count"] >= 3)
    ].head(20)
    print("\nPROVISIONAL candidates (at least 15 metadata instances and 3 signers):")
    print("Video files and URLs have NOT been verified; these are not usable-video counts.")
    print(candidates.to_string(index=False) if not candidates.empty
          else "No glosses meet the metadata thresholds.")
    print(f"\nCSV files saved in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

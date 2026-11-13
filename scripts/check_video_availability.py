r"""Check local WLASL videos without downloading or contacting source URLs.

Run from the project root:
    .\.venv\Scripts\python.exe scripts/check_video_availability.py

Matches video filenames by exact video_id (for example, 69241.mp4), recursively
under dataset/. File existence does not guarantee that a video can be decoded.
Overwrites dataset/metadata/video_availability_report.csv.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORDS = ["book", "drink", "computer", "before", "chair", "go", "clothes",
         "who", "help", "yes"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg",
                    ".mpeg", ".wmv", ".flv"}


def check_availability(metadata, dataset_dir):
    """Keep one report row per selected metadata instance, including duplicates."""
    if "gloss" not in metadata.columns:
        raise ValueError("Input CSV is missing the 'gloss' column; cannot select words.")

    columns = ["gloss", "video_id", "signer_id", "split", "url"]
    selected = metadata.loc[metadata["gloss"].isin(WORDS)].copy()
    for column in columns:
        if column not in selected.columns:
            print(f"WARNING: Missing '{column}' column; using blank values.",
                  file=sys.stderr)
            selected[column] = ""
    selected = selected[columns].fillna("")

    # shortcut: videos must be named by video_id; add a mapping for renamed files.
    videos = {}
    for path in sorted(dataset_dir.rglob("*")):
        if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
            videos.setdefault(path.stem, path)

    paths = []
    statuses = []
    for video_id in selected["video_id"]:
        path = videos.get(video_id) if video_id else None
        paths.append(str(path.resolve()) if path else "")
        statuses.append("found" if path else
                        "missing_video_id" if not video_id else "missing")

    selected["local_file_status"] = statuses
    selected["local_file_path"] = paths
    selected["remote_status"] = "not_checked"
    return selected


def main():
    dataset_dir = PROJECT_ROOT / "dataset"
    input_path = dataset_dir / "metadata" / "wlasl_instances.csv"
    output_path = dataset_dir / "metadata" / "video_availability_report.csv"
    try:
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Metadata CSV not found: {input_path}. "
                "Run scripts/explore_dataset.py first."
            )
        # Read IDs as text to preserve leading zeros and leave missing fields blank.
        metadata = pd.read_csv(input_path, dtype=str, keep_default_na=False)
        report = check_availability(metadata, dataset_dir)
        report.to_csv(output_path, index=False)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    summary = pd.DataFrame(index=pd.Index(WORDS, name="gloss"))
    summary["metadata_instances"] = report.groupby("gloss").size()
    summary["locally_found"] = (
        report.loc[report["local_file_status"].eq("found")].groupby("gloss").size()
    )
    summary = summary.fillna(0).astype(int)
    summary["missing_local"] = summary["metadata_instances"] - summary["locally_found"]
    print(summary.to_string())
    print("\nCounts refer to metadata instances, not unique video files.")
    print("Missing local counts include instances without a video_id.")
    print("Remote status: not_checked. No URLs contacted or videos downloaded.")
    print(f"Report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

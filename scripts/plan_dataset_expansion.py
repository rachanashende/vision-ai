r"""Plan WLASL expansion without network requests, downloads, or data changes.

PowerShell: python scripts/plan_dataset_expansion.py --dry-run
Dry-run is the only mode and the default. --overwrite permits replacing only
this planner's CSV and summary. No download command is implemented here.
Initial train target: 15 valid clips/class, with 20 as a stretch target, plus
available original validation/test instances. These are planning milestones,
not availability guarantees or evidence of a reliable classifier.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import pandas as pd

from download_pilot_videos import WORDS, source_problem
from explore_dataset import find_metadata, flatten_entries
from inspect_videos import inspect_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
TRAIN_MIN, TRAIN_STRETCH = 15, 20
IDENTITY_FIELDS = ("gloss", "video_id", "split", "signer_id", "url")
PLAN_COLUMNS = ["gloss", "video_id", "split", "signer_id", "source", "url", "frame_start",
                "frame_end", "fps", "local_status", "local_path", "local_errors", "download_status",
                "pilot_status", "pilot_error", "remote_status", "source_support", "source_notes",
                "metadata_completeness", "signer_new_for_split", "priority", "priority_reason",
                "recommended_for_initial_pilot", "destination"]


def reconcile_metadata(original_rows, instances):
    """Original JSON is authoritative; stop on duplicate IDs or identity conflicts."""
    if not set(IDENTITY_FIELDS).issubset(instances.columns):
        raise ValueError("Instances CSV is missing required identity columns.")
    rows = [{key: "" if value is None else str(value) for key, value in row.items()}
            for row in original_rows if row["gloss"] in WORDS]
    selected = instances.loc[instances["gloss"].isin(WORDS)].fillna("")
    ids = [row["video_id"] for row in rows]
    if len(ids) != len(set(ids)) or selected["video_id"].duplicated().any():
        raise ValueError("Duplicate video IDs in selected metadata; resolve ambiguity first.")
    csv_rows = {row["video_id"]: row for row in selected.to_dict("records")}
    if set(ids) != set(csv_rows):
        raise ValueError("Original JSON and instances CSV contain different selected video IDs.")
    for row in rows:
        for field in IDENTITY_FIELDS:
            if row.get(field, "") != str(csv_rows[row["video_id"]][field]):
                raise ValueError(f"JSON/CSV conflict for {row['video_id']}: {field}. No split or signer assignments changed.")
    return rows


def inventory_local(rows, raw_dir):
    """Fully decode existing files; never trust old download success as local validity."""
    by_id = defaultdict(list)
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".mp4":
            by_id[path.stem].append(path)
    inventory = {}
    for row in rows:
        matches = by_id[row["video_id"]]
        matching_gloss = [path for path in matches if len(path.relative_to(raw_dir).parts) > 1
                          and path.relative_to(raw_dir).parts[0] == row["gloss"]]
        result = dict(local_status="missing", local_path="", local_errors="")
        if matches and not matching_gloss:
            result.update(local_status="misplaced_file", local_errors="Video ID exists outside its expected gloss directory; inspect manually.")
        for path in matching_gloss:
            inspected = inspect_video(path)
            result.update(local_status="valid" if inspected["valid"] else "invalid",
                          local_path=str(path.resolve()), local_errors=inspected["errors"])
            if inspected["valid"]:
                break
        inventory[row["video_id"]] = result
    return inventory


def build_plan(rows, inventory, pilot_report, raw_dir):
    history = {}
    if not pilot_report.empty:
        if not {"video_id", "gloss", "url", "status"}.issubset(pilot_report.columns):
            raise ValueError("Pilot report is missing required columns.")
        if pilot_report["video_id"].duplicated().any():
            raise ValueError("Duplicate video IDs in pilot report; historical availability is ambiguous.")
        history = {row["video_id"]: row for row in pilot_report.to_dict("records")}
    valid_counts = Counter()
    signers = defaultdict(set)
    for row in rows:
        key = (row["gloss"], row["split"])
        if inventory[row["video_id"]]["local_status"] == "valid":
            valid_counts[key] += 1
            if row.get("signer_id"):
                signers[key].add(row["signer_id"])
    planned_counts = Counter()
    pending, existing = [], []
    for metadata in rows:
        row = {key: metadata.get(key, "") for key in
               ("gloss", "video_id", "split", "signer_id", "source", "url", "frame_start", "frame_end", "fps")}
        row.update(inventory[row["video_id"]])
        previous = history.get(row["video_id"], {})
        if previous.get("url") != row["url"] or previous.get("gloss") != row["gloss"]:
            previous = {}
        row.update(pilot_status=previous.get("status", ""), pilot_error=previous.get("error", ""),
                   remote_status="not_checked", destination=str(raw_dir / row["gloss"] / f"{row['video_id']}.mp4"))
        problem = source_problem(row["url"])
        row["source_support"] = problem[0] if problem else "supported_direct_mp4"
        row["source_notes"] = problem[1] if problem else "URL syntax is supported; remote availability untested."
        prior_failure = row["pilot_status"] in ("failed", "rejected", "access_refused", "restricted_source", "existing_invalid")
        if row["pilot_status"] in ("downloaded", "already_valid"):
            row["remote_status"] = "historical_success_not_rechecked"
        elif prior_failure:
            row["remote_status"] = "previous_failure_not_rechecked"
        fields = ("split", "signer_id", "url", "source", "frame_start", "frame_end", "fps")
        row["metadata_completeness"] = sum(row.get(field, "") != "" for field in fields)
        row["recommended_for_initial_pilot"] = False
        row["priority"] = ""
        row["signer_new_for_split"] = False
        complete = row["split"] in SPLITS and bool(row["signer_id"]) and bool(row["url"])
        row["actionable"] = (row["local_status"] == "missing" and complete and not problem and not prior_failure)
        if row["local_status"] == "valid":
            row.update(download_status="already_local_valid", priority_reason="Already valid locally; no download needed.")
            existing.append(row)
        else:
            row["download_status"] = ("candidate_not_downloaded" if row["actionable"] else "manual_source_or_file_review")
            pending.append(row)
    ranked = []
    while pending:
        def rank(row):
            key = (row["gloss"], row["split"])
            coverage = valid_counts[key] + planned_counts[key]
            phase = 0 if row["split"] in ("val", "test") else 1
            if row["split"] == "train" and coverage >= TRAIN_STRETCH:
                phase = 2
            novelty = bool(row["signer_id"]) and row["signer_id"] not in signers[key]
            history_rank = 0 if row["remote_status"] == "historical_success_not_rechecked" else 1
            return (not row["actionable"], phase, coverage > 0, -row["metadata_completeness"],
                    not novelty, history_rank, WORDS.index(row["gloss"]), row["video_id"])
        row = min(pending, key=rank)
        pending.remove(row)
        key = (row["gloss"], row["split"])
        row["priority"] = len(ranked) + 1
        row["signer_new_for_split"] = bool(row["signer_id"]) and row["signer_id"] not in signers[key]
        recommend = row["actionable"] and (row["split"] in ("val", "test") or
                    valid_counts[key] + planned_counts[key] < TRAIN_STRETCH)
        row["recommended_for_initial_pilot"] = bool(recommend)
        row["priority_reason"] = (
            "Original validation/test coverage first" if row["split"] in ("val", "test") else
            "Additional original training samples; aim for 15, stretch to 20"
        ) + f"; completeness {row['metadata_completeness']}/7; new signer for split={row['signer_new_for_split']}"
        if not row["actionable"]:
            row["priority_reason"] = "Manual review required before any future request; " + row["priority_reason"]
        if recommend:
            planned_counts[key] += 1
            signers[key].add(row["signer_id"])
        ranked.append(row)
    return pd.DataFrame(ranked + existing).reindex(columns=PLAN_COLUMNS)


def coverage_summary(plan):
    rows = []
    for word in WORDS:
        for split in SPLITS:
            group = plan.loc[plan["gloss"].eq(word) & plan["split"].eq(split)]
            valid = int(group["local_status"].eq("valid").sum())
            rows.append(dict(gloss=word, split=split, metadata_instances=len(group), valid_local=valid,
                             missing_or_invalid=len(group) - valid,
                             candidates=int(group["download_status"].eq("candidate_not_downloaded").sum()),
                             recommended=int(group["recommended_for_initial_pilot"].sum()),
                             gap_to_15=max(0, TRAIN_MIN - valid) if split == "train" else "",
                             gap_to_20=max(0, TRAIN_STRETCH - valid) if split == "train" else ""))
    return pd.DataFrame(rows)


def check_outputs(paths, overwrite):
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite only to replace planner outputs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Plan only (default and only mode); no network or downloads.")
    parser.add_argument("--overwrite", action="store_true", help="Replace this planner's CSV and summary only.")
    args = parser.parse_args()
    output = PROJECT_ROOT / "dataset/metadata/dataset_expansion_plan.csv"
    summary_path = PROJECT_ROOT / "results/evaluation/dataset_expansion_summary.md"
    try:
        check_outputs([output, summary_path], args.overwrite)
        original_path = PROJECT_ROOT / "dataset/WLASL/start_kit/WLASL_v0.3.json"
        if not original_path.is_file():
            original_path = find_metadata(PROJECT_ROOT / "dataset/WLASL")
        original, _ = flatten_entries(json.loads(original_path.read_text(encoding="utf-8-sig")))
        instances = pd.read_csv(PROJECT_ROOT / "dataset/metadata/wlasl_instances.csv", dtype=str, keep_default_na=False)
        rows = reconcile_metadata(original, instances)
        pilot = pd.read_csv(PROJECT_ROOT / "dataset/metadata/pilot_download_report.csv", dtype=str, keep_default_na=False)
        raw_dir = PROJECT_ROOT / "dataset/raw_videos"
        print("DRY RUN: local decoding and planning only; no network or downloads.")
        inventory = inventory_local(rows, raw_dir)
        plan = build_plan(rows, inventory, pilot, raw_dir)
        coverage = coverage_summary(plan)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.overwrite else "x", encoding="utf-8", newline="") as stream:
            plan.to_csv(stream, index=False)
        lines = ["# WLASL dataset expansion plan", "", f"Original metadata: `{original_path}`.",
                 "The original JSON and instances CSV were cross-checked for identity, split, signer, and URL consistency.",
                 "All source train/val/test assignments and signer IDs are preserved; IDs are read as text.", "",
                 "Initial planning target: 15 valid training clips per class, with 20 as a stretch target, plus",
                 "available original validation/test clips. These counts are neither guaranteed nor sufficient",
                 "evidence of a reliable classifier. Some classes have only 15-19 training metadata instances,",
                 "so a target of 20 cannot be reached without changing scope. Splits must not be reassigned.", "",
                 "Local validity was checked by full OpenCV decoding using the existing inspector. A local",
                 "validity result does not verify sign correctness, sign boundaries, trimming, or training quality.",
                 "No boundary reviews, provisional retention decisions, videos, NPZs, or existing reports were changed.", "",
                 "No network requests were made. Supported URL syntax is not proof of availability. Previous",
                 "pilot success/failure is historical evidence for the exact same URL, never a live availability claim.",
                 "Restricted/unsupported URLs, failed prior requests, incomplete identity metadata, and existing",
                 "invalid/misplaced files require manual review and are not recommended automatically.",
                 "The existing pilot downloader supports public direct MP4s only; YouTube/SWF require a separately",
                 "agreed supported workflow. No access controls are bypassed and no dependencies are installed.", "",
                 "Ranked candidates prioritize actionable original validation/test instances, then training gaps.",
                 "Within those phases, empty split coverage, metadata completeness, and new signer coverage",
                 "are prioritized, followed by historical URL success. Signer novelty is tracked separately",
                 "within each original class/split; this does not establish signer-disjoint splits.",
                 "All 277 selected metadata instances are retained in the plan when present in this dataset;",
                 "existing valid videos have blank download priority. Lower numbers mean earlier candidates.",
                 "Candidate counts below are actionable but untested remote requests; missing/invalid counts",
                 "also include unsupported and previously failed sources. Recommendations are a queue, not a promise.", "",
                 "| Gloss | Split | Metadata | Valid local | Missing/invalid | Untested candidates | Recommended | Gap to 15 train | Gap to 20 train |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in coverage.to_dict("records"):
            lines.append("| " + " | ".join(str(row[key]) for key in coverage.columns) + " |")
        lines += ["", "A separate explicit download request is required before implementing or running expansion",
                  "downloads. Do not use the old pilot command as an expansion downloader: it selects at most",
                  "three videos per class and does not consume this plan. This planner has no download mode.", ""]
        with summary_path.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
        print(coverage.to_string(index=False))
        print(f"\nPlan: {output}\nSummary: {summary_path}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

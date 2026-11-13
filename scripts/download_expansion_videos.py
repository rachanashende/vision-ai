r"""Resume a small expansion batch; dry-run is the default and never uses network.

PowerShell, from the project root:
    python scripts/download_expansion_videos.py --dry-run --limit 10
    python scripts/download_expansion_videos.py --download --limit 10

Only recommended candidate_not_downloaded rows from the existing plan are used.
Validation/test precede train. Existing valid files and known HTTP 404/410s do
not consume the limit, so later runs progress through the remaining queue.
--retries counts EXTRA transient-error attempts (default 1, maximum 3).
--timeout is a per-socket timeout in seconds; transfers also have a time budget.
Every attempt and skip/preview event is appended to the report with a run ID.
No automatic installs, metadata edits, cookies, redirects, or access bypasses.
Run one downloader process at a time; the CSV history is not a multiwriter log.
"""

import argparse
import csv
from datetime import datetime, timezone
from http.client import HTTPException
import math
from pathlib import Path
import re
import socket
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener
from uuid import uuid4

import cv2
import pandas as pd

from download_pilot_videos import MAX_BYTES, NoRedirects, WORDS, source_problem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_COLUMNS = ["run_id", "timestamp_utc", "gloss", "video_id", "split", "signer_id",
                  "url", "source", "attempt", "status", "http_status", "error_type",
                  "error", "saved_path", "destination"]


def validate_video(path):
    """Validate a .part or final file without changing it."""
    if not path.is_file() or path.stat().st_size == 0:
        return False, "Missing or empty video."
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False, "OpenCV could not open the file."
        count, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(count) or count <= 0 or not math.isfinite(fps) or fps <= 0:
            return False, "Video must have a positive finite frame count and FPS."
        success, frame = capture.read()
        if not success or frame is None or frame.size == 0:
            return False, "OpenCV could not decode a frame."
        decoded = 1
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None or frame.size == 0:
                return False, "Decoder returned an empty frame."
            decoded += 1
        if decoded < int(count):
            return False, f"Incomplete/undecodable video: decoded {decoded} of {int(count)} reported frames."
        return True, ""
    except cv2.error as error:
        return False, str(error)
    finally:
        capture.release()


def read_history(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != REPORT_COLUMNS:
            raise ValueError("Existing expansion report schema differs; preserve and inspect it before continuing.")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Existing report contains an incomplete row; inspect it before appending.")
    return rows


def append_report(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in REPORT_COLUMNS})


def select_candidates(plan, split):
    required = {"gloss", "video_id", "split", "signer_id", "url", "source", "priority",
                "recommended_for_initial_pilot", "download_status", "source_support"}
    if not required.issubset(plan.columns):
        raise ValueError("Expansion plan is missing required columns.")
    if plan["video_id"].duplicated().any():
        raise ValueError("Duplicate video IDs in expansion plan.")
    flags = plan["recommended_for_initial_pilot"].str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Invalid recommended_for_initial_pilot Boolean values.")
    selected = plan.loc[flags.eq("true") & plan["download_status"].eq("candidate_not_downloaded")].copy()
    if split != "all":
        selected = selected.loc[selected["split"].eq(split)]
    for row in selected.to_dict("records"):
        if row["gloss"] not in WORDS or not re.fullmatch(r"[0-9]+", row["video_id"]):
            raise ValueError("Invalid gloss/video_id in candidate plan.")
        if row["split"] not in ("train", "val", "test") or not row["signer_id"].strip():
            raise ValueError("Missing candidate split or signer metadata.")
    selected["split_order"] = selected["split"].map({"val": 0, "test": 0, "train": 1})
    selected["priority_order"] = pd.to_numeric(selected["priority"], errors="raise")
    return selected.sort_values(["split_order", "priority_order", "video_id"]).to_dict("records")


def known_permanent_failures(history):
    latest = {}
    for row in history:
        if int(row["attempt"]) > 0:
            latest[(row["gloss"], row["video_id"], row["url"])] = row["http_status"]
    return {key for key, status in latest.items() if status in ("404", "410")}


def transfer_once(url, destination, timeout):
    """One request; clean temporary bytes even on interruption. Never overwrite."""
    result = dict(status="failed", http_status="", error_type="", error="", saved_path="")
    temporary = None
    retryable = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.stem}-",
                                         suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            request = Request(url, headers={"User-Agent": "ASL-Expansion-Downloader/1.0"})
            started = time.monotonic()
            with build_opener(NoRedirects()).open(request, timeout=timeout) as response:
                result["http_status"] = str(response.status)
                if response.status != 200:
                    raise ValueError(f"Expected complete HTTP 200 response; received {response.status}.")
                if response.headers.get_content_type() not in ("video/mp4", "application/octet-stream", "binary/octet-stream"):
                    raise ValueError("Response is not MP4/binary video content; page/error response rejected.")
                length = response.headers.get("Content-Length")
                expected = int(length) if length is not None else None
                if expected is not None and not 0 < expected <= MAX_BYTES:
                    raise ValueError("Invalid/oversized Content-Length (limit 256 MiB).")
                received = 0
                while True:
                    if time.monotonic() - started > max(120, timeout * 4):
                        raise TimeoutError("Transfer time budget exceeded.")
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    if received == 0 and chunk.lstrip().lower().startswith((b"<!doctype", b"<html", b"<?xml", b"{", b"[")):
                        raise ValueError("HTML/JSON/XML error body rejected.")
                    received += len(chunk)
                    if received > MAX_BYTES:
                        raise ValueError("Transfer exceeds 256 MiB limit.")
                    stream.write(chunk)
                if expected is not None and received != expected:
                    raise OSError("Incomplete download: Content-Length mismatch.")
        valid, error = validate_video(temporary)
        if not valid:
            raise ValueError(f"Downloaded file rejected: {error}")
        # Same-volume, atomic publication of validated bytes; fails if target exists.
        destination.hardlink_to(temporary)
        result.update(status="downloaded", saved_path=str(destination))
    except HTTPError as error:
        result.update(status="http_error", http_status=str(error.code), error_type="HTTPError", error=str(error))
        retryable = 500 <= error.code < 600
        error.close()
    except ValueError as error:
        result.update(status="rejected", error_type="invalid_download", error=str(error))
    except (PermissionError, FileExistsError) as error:
        result.update(status="local_file_error", error_type=type(error).__name__, error=str(error))
    except (OSError, URLError, HTTPException) as error:
        reason = error.reason if isinstance(error, URLError) else error
        kind = "dns_error" if isinstance(reason, socket.gaierror) else "timeout" if isinstance(reason, TimeoutError) else type(error).__name__
        result.update(error_type=kind, error=str(error))
        retryable = True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result, retryable


def run_batch(candidates, history, raw_dir, report_path, download=False, limit=10,
              timeout=30, retries=1, retry_404s=False):
    permanent = known_permanent_failures(history)
    run_id = str(uuid4())
    processed = 0
    failures = 0
    for row in candidates:
        if processed >= limit:
            break
        destination = raw_dir / row["gloss"] / f"{row['video_id']}.mp4"
        base = {key: row[key] for key in ("gloss", "video_id", "split", "signer_id", "url", "source")}
        base.update(run_id=run_id, destination=str(destination))

        def emit(result, attempt=0):
            event = {**base, **result, "attempt": attempt,
                     "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            append_report(report_path, event)
            print(f"{row['gloss']}/{row['video_id']} [{row['split']}] attempt={attempt}: "
                  f"{result['status']}" + (f" - {result.get('error', '')}" if result.get("error") else ""))

        if destination.exists():
            try:
                valid, error = validate_video(destination)
            except (OSError, cv2.error) as exception:
                valid, error = False, str(exception)
            emit(dict(status="already_valid" if valid else "existing_invalid",
                      saved_path=str(destination) if valid else "", error=error))
            if not valid:
                failures += 1
            continue
        if (row["gloss"], row["video_id"], row["url"]) in permanent and not retry_404s:
            emit(dict(status="known_permanent_skipped", error="Previous HTTP 404/410 for this exact URL; use --retry-404s to request a new attempt."))
            continue
        processed += 1
        problem = source_problem(row["url"])
        if problem or row["source_support"] != "supported_direct_mp4":
            emit(dict(status="unsupported_or_restricted", error=problem[1] if problem else "Plan source is unsupported."))
            failures += 1
            continue
        if not download:
            emit(dict(status="dry_run", error="No request made; remote availability unknown."))
            print(f"  URL: {row['url']}\n  Destination: {destination}")
            continue
        for attempt in range(1, retries + 2):
            result, retryable = transfer_once(row["url"], destination, timeout)
            emit(result, attempt)
            if not retryable or attempt == retries + 1:
                failures += result["status"] != "downloaded"
                break
            time.sleep(2 * attempt)
        time.sleep(1)
    print(f"Run {run_id}: {processed} candidates considered (limit {limit}); report history appended to {report_path}")
    return failures


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def retry_count(value):
    number = int(value)
    if not 0 <= number <= 3:
        raise argparse.ArgumentTypeError("--retries must be between 0 and 3 extra attempts.")
    return number


def positive_timeout(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("--timeout must be positive and finite.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only, append preview history (default).")
    mode.add_argument("--download", action="store_true", help="Permit actual network requests for this limited batch.")
    parser.add_argument("--limit", type=positive_int, default=10)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--timeout", type=positive_timeout, default=30)
    parser.add_argument("--retries", type=retry_count, default=1)
    parser.add_argument("--retry-404s", action="store_true", help="Allow new requests for historically recorded 404/410 URLs.")
    args = parser.parse_args()
    report_path = PROJECT_ROOT / "dataset/metadata/expansion_download_report.csv"
    try:
        history = read_history(report_path)
        plan = pd.read_csv(PROJECT_ROOT / "dataset/metadata/dataset_expansion_plan.csv", dtype=str, keep_default_na=False)
        candidates = select_candidates(plan, args.split)
        original = pd.read_csv(PROJECT_ROOT / "dataset/metadata/wlasl_instances.csv", dtype=str, keep_default_na=False)
        if original["video_id"].duplicated().any():
            raise ValueError("Duplicate IDs in original instances metadata.")
        by_id = original.set_index("video_id").to_dict("index")
        for row in candidates:
            reference = by_id.get(row["video_id"])
            if reference is None or any(reference[key] != row[key] for key in ("gloss", "split", "signer_id", "url")):
                raise ValueError(f"Plan/original metadata conflict for {row['video_id']}; no requests made.")
        print("DOWNLOAD MODE" if args.download else "DRY RUN: no network requests or downloads")
        failures = run_batch(candidates, history, PROJECT_ROOT / "dataset/raw_videos", report_path,
                             args.download, args.limit, args.timeout, args.retries, args.retry_404s)
        return 1 if failures else 0
    except (OSError, ValueError, KeyError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

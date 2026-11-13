r"""Plan or download a small WLASL pilot without changing the environment.

PowerShell, from the project root:
    .\.venv\Scripts\python.exe scripts/download_pilot_videos.py --dry-run
    .\.venv\Scripts\python.exe scripts/download_pilot_videos.py --download

Default: dry-run, with no network access. Both modes overwrite the CSV report.
Requires existing pandas and OpenCV; uses urllib for public direct MP4 URLs.
WLASL guidance: https://github.com/dxli94/WLASL#download-original-videos
YouTube needs yt-dlp; SWF needs conversion tools such as FFmpeg. This script
reports those sources as unsupported and does not install or invoke those tools.
Raw videos are not trimmed to WLASL frame_start/frame_end annotations.
"""

import argparse
from http.client import HTTPException
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORDS = ["book", "drink", "computer", "before", "chair", "go", "clothes",
         "who", "help", "yes"]
REPORT_COLUMNS = ["gloss", "video_id", "split", "url", "destination", "status", "error"]
MAX_BYTES = 256 * 1024 * 1024


def source_problem(url):
    """Classify URLs locally; never probe a server during selection or dry-run."""
    try:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return "unsupported_source", "Missing or invalid HTTP(S) URL."
        path = unquote(parts.path).lower()
        if parts.username or parts.password or any(
            segment in path.split("/")
            for segment in ("protected_media", "protected", "private", "login")
        ):
            return "restricted_source", "URL indicates restricted access; skipped."
        host = parts.hostname.lower()
        if host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com"):
            return "unsupported_source", "YouTube requires yt-dlp; not used by this script."
        if not path.endswith(".mp4"):
            return "unsupported_source", "Only direct MP4 URLs are supported; no page extraction or conversion."
    except ValueError as error:
        return "unsupported_source", str(error)
    return None


def select_pilot(metadata):
    """Prefer train, then val/test, and supported sources within the same split."""
    for column in ("gloss", "video_id"):
        if column not in metadata:
            raise ValueError(f"Metadata CSV is missing required column: {column}")
    metadata = metadata.copy().fillna("")
    for column in ("split", "url"):
        if column not in metadata:
            print(f"WARNING: Missing {column} column; using blank values.", file=sys.stderr)
            metadata[column] = ""
    selected = []
    seen_ids = set()
    for word in WORDS:
        rows = metadata.loc[metadata["gloss"].eq(word)].to_dict("records")
        rows.sort(key=lambda row: (
            {"train": 0, "val": 1, "test": 2}.get(row["split"], 3),
            source_problem(row["url"]) is not None,
        ))
        count = 0
        for row in rows:
            video_id = row["video_id"]
            if not re.fullmatch(r"[0-9]+", video_id):
                print(f"WARNING: {word}: missing/invalid video_id {video_id!r}; skipped.",
                      file=sys.stderr)
                continue
            if video_id in seen_ids:
                continue
            seen_ids.add(video_id)
            selected.append(row)
            count += 1
            if count == 3:
                break
        if count < 3:
            print(f"WARNING: Only {count} distinct video IDs selected for {word}.")
    return selected


def valid_video(path):
    """Require a nonempty file that opens and decodes at least one frame."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        success, frame = capture.read()
        return bool(success and frame is not None and frame.size > 0)
    except cv2.error:
        return False
    finally:
        capture.release()


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise ValueError("Redirect refused; review the destination manually.")


def download_video(url, destination):
    """Two attempts maximum; only publish a completed, decoded temporary file."""
    if destination.exists():
        if valid_video(destination):
            return "already_valid", ""
        return "existing_invalid", "Existing file cannot decode; left untouched. Move it before retrying."
    problem = source_problem(url)
    if problem:
        return problem
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = build_opener(NoRedirects())
    for attempt in range(2):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f".{destination.stem}-",
                suffix=".partial.mp4", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                request = Request(url, headers={"User-Agent": "ASL-Pilot-Downloader/1.0"})
                started = time.monotonic()
                with opener.open(request, timeout=30) as response:
                    if response.headers.get_content_type() in ("text/html", "application/json"):
                        raise ValueError("Server returned a page/JSON instead of video data.")
                    expected_size = response.headers.get("Content-Length")
                    expected_size = int(expected_size) if expected_size is not None else None
                    if expected_size is not None and expected_size > MAX_BYTES:
                        raise ValueError("Video exceeds the 256 MiB pilot limit.")
                    received = 0
                    while True:
                        if time.monotonic() - started > 120:
                            raise TimeoutError("Download exceeded the 120-second transfer budget.")
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > MAX_BYTES:
                            raise ValueError("Video exceeds the 256 MiB pilot limit.")
                        stream.write(chunk)
                    if expected_size is not None and received != expected_size:
                        raise OSError("Incomplete download: Content-Length does not match.")
            if not valid_video(temporary):
                raise ValueError("OpenCV could not open and decode a frame; download rejected.")
            # A hard link publishes without replacing even a concurrently created file.
            destination.hardlink_to(temporary)
            return "downloaded", ""
        except HTTPError as error:
            details = f"HTTP {error.code}: {error.reason}"
            if error.code in (401, 403, 429):
                return "access_refused", details + "; no bypass or further requests."
            if error.code < 500:
                return "failed", details
        except ValueError as error:
            return "rejected", str(error)
        except (OSError, URLError, HTTPException) as error:
            details = str(error)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if attempt == 0:
            time.sleep(1)
    return "failed", f"Two attempts failed: {details}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only; no network (default).")
    mode.add_argument("--download", action="store_true", help="Download supported public MP4 URLs.")
    args = parser.parse_args()
    input_path = PROJECT_ROOT / "dataset/metadata/wlasl_instances.csv"
    report_path = PROJECT_ROOT / "dataset/metadata/pilot_download_report.csv"
    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing {input_path}. Run scripts/explore_dataset.py first.")
        metadata = pd.read_csv(input_path, dtype=str, keep_default_na=False)
        selected = select_pilot(metadata)
        report = []
        print("DOWNLOAD MODE" if args.download else "DRY RUN: no network access or downloads")
        print("Raw files are not trimmed; retain WLASL frame_start/frame_end for preprocessing.")
        for row in selected:
            destination = PROJECT_ROOT / "dataset/raw_videos" / row["gloss"] / f"{row['video_id']}.mp4"
            if args.download:
                try:
                    status, error = download_video(row["url"], destination)
                except OSError as exception:
                    status, error = "failed", str(exception)
            else:
                problem = source_problem(row["url"])
                status, error = problem if problem else ("dry_run", "Not requested; remote availability unknown.")
            record = {key: row[key] for key in ("gloss", "video_id", "split", "url")}
            record.update(destination=str(destination), status=status, error=error)
            report.append(record)
            print(f"{row['gloss']} | {row['video_id']} | {row['split']} | {status}\n"
                  f"  URL: {row['url']}\n  Destination: {destination}")
            if error:
                print(f"  Details: {error}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(report, columns=REPORT_COLUMNS).to_csv(report_path, index=False)
            if args.download:
                time.sleep(1)  # Follow WLASL guidance to avoid spamming source hosts.
        if not report:
            pd.DataFrame(columns=REPORT_COLUMNS).to_csv(report_path, index=False)
        print(f"Selected {len(report)} videos (maximum 30). Report: {report_path}")
        return 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

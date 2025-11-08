#!/usr/bin/env python3
"""
Resumable downloader for Qlib dataset archives with unzip.

Usage examples:
  python scripts/qlib_resumable_download.py --region cn --interval 1d \
      --target_dir ~/.qlib/qlib_data/cn_data --delete-old false --exists-skip true

- Resumes from partial .part file
- Falls back to fresh download if server doesn't support Range
- Unzips into target_dir without interactive prompts
"""
import argparse
import os
import sys
import time
import zipfile
import shutil
from pathlib import Path
from typing import Optional

import requests

DEFAULT_BASE = "https://github.com/SunsetWolf/qlib_dataset/releases/download"


def build_url(region: str, interval: str, version: str) -> str:
    # Always use latest for robustness; version can be overridden
    file_name = f"qlib_data_{region.lower()}_{interval.lower()}_latest.zip"
    return f"{DEFAULT_BASE}/{version}/{file_name}"


def resumable_download(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    headers = {}
    pos = 0

    # Probe remote size
    head = requests.head(url, timeout=30, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    accept_range = head.headers.get("Accept-Ranges", "").lower() == "bytes"

    if part_path.exists():
        pos = part_path.stat().st_size
        # If partial larger than remote (rare), restart
        if total and pos > total:
            part_path.unlink()
            pos = 0

    # Prepare request
    if accept_range and pos > 0:
        headers["Range"] = f"bytes={pos}-"

    # Stream download
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code not in (200, 206):
            r.raise_for_status()
        mode = "ab" if r.status_code == 206 and pos > 0 else "wb"
        if mode == "wb" and part_path.exists():
            part_path.unlink()

        downloaded = pos
        last_log = time.time()
        with open(part_path, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                # Minimal console progress (no tqdm to avoid heavy deps)
                now = time.time()
                if now - last_log >= 1:
                    if total:
                        pct = downloaded / total * 100
                        speed = len(chunk) / max(now - last_log, 1)
                        print(f"Downloading: {downloaded}/{total} ({pct:.2f}%)", flush=True)
                    else:
                        print(f"Downloading: {downloaded} bytes", flush=True)
                    last_log = now

    # Finalize file
    if out_path.exists():
        out_path.unlink()
    part_path.rename(out_path)
    return out_path


def unzip_to_dir(zip_path: Path, target_dir: Path, delete_old: bool = False):
    target_dir.mkdir(parents=True, exist_ok=True)

    if delete_old:
        rm_dirs = ["features", "calendars", "instruments", "features_cache", "dataset_cache"]
        for name in rm_dirs:
            p = target_dir / name
            if p.exists():
                print(f"Deleting: {p}")
                shutil.rmtree(p)

    with zipfile.ZipFile(str(zip_path), "r") as zp:
        for member in zp.namelist():
            zp.extract(member, str(target_dir))


def data_exists(target_dir: Path) -> bool:
    # Heuristic: must have instruments and calendars directories populated
    return (target_dir / "instruments").exists() and (target_dir / "calendars").exists()


def main():
    ap = argparse.ArgumentParser(description="Resumable Qlib data downloader")
    ap.add_argument("--target_dir", default="~/.qlib/qlib_data/cn_data")
    ap.add_argument("--region", default="cn", choices=["cn", "us"]) 
    ap.add_argument("--interval", default="1d", choices=["1d", "1min"]) 
    ap.add_argument("--version", default="v2")
    ap.add_argument("--delete-old", dest="delete_old", default=False, action="store_true")
    ap.add_argument("--exists-skip", dest="exists_skip", default=True, action="store_true")
    ap.add_argument("--no-exists-skip", dest="exists_skip", action="store_false")
    ap.add_argument("--keep-zip", dest="keep_zip", default=False, action="store_true")
    args = ap.parse_args()

    target_dir = Path(args.target_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.exists_skip and data_exists(target_dir):
        print(f"Data already exists in {target_dir}, skipping download.")
        return 0

    url = build_url(args.region, args.interval, args.version)
    out_zip = target_dir / f"qlib_data_{args.region}_{args.interval}_latest.zip"

    print(f"Downloading from: {url}")
    print(f"Saving to: {out_zip}.part (resumable)")

    try:
        zip_path = resumable_download(url, out_zip)
        print(f"Download completed: {zip_path}")
        print("Unzipping...")
        unzip_to_dir(zip_path, target_dir, delete_old=args.delete_old)
        print("Unzip completed.")
        if not args.keep_zip:
            zip_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

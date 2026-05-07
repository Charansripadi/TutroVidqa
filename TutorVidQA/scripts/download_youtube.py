"""
Download educational videos from YouTube using yt-dlp.

Saves:
- datasets/raw_videos/<video_id>.mp4
- datasets/metadata/videos.jsonl (append-only)
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List

from yt_dlp import YoutubeDL

from src.common import append_jsonl, ensure_dir, setup_logging


def download(url: str, out_dir: str) -> Dict:
    """
    Download a single YouTube URL to `out_dir` and return metadata.

    Key steps:
    - extract info (video id/title/channel/duration)
    - download best mp4 (or best available) into raw_videos/
    """
    ensure_dir(out_dir)
    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("uploader") or info.get("channel"),
        "duration_s": info.get("duration"),
        "url": url,
        "filepath": os.path.join(out_dir, f"{info.get('id')}.mp4"),
    }


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="YouTube video URL")
    p.add_argument("--out_dir", default="datasets/raw_videos", help="Raw video folder")
    p.add_argument("--meta_jsonl", default="datasets/metadata/videos.jsonl")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)

    logging.info("Downloading %s", args.url)
    meta = download(args.url, args.out_dir)
    append_jsonl(args.meta_jsonl, [meta])
    logging.info("Saved %s", meta["filepath"])


if __name__ == "__main__":
    main()


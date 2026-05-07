"""
Segment videos into fixed windows + extract keyframes.

Writes:
- datasets/segments/<video_id>/segments.json
- datasets/segments/<video_id>/keyframes/segment_000000.jpg ...
"""

from __future__ import annotations

import argparse
import logging
import os
from glob import glob

from src.common import ensure_dir, setup_logging, write_json
from src.video_segmenter import segment_video


def _video_id(path: str) -> str:
    """Infer video_id from filename (without extension)."""
    return os.path.splitext(os.path.basename(path))[0]


def segment_one(video_path: str, out_root: str, segment_s: int) -> str:
    """Segment a single video and return output segments.json path."""
    vid = _video_id(video_path)
    out_dir = ensure_dir(os.path.join(out_root, vid))
    frames_dir = ensure_dir(os.path.join(out_dir, "keyframes"))
    segs = segment_video(video_path, segment_duration_s=segment_s, frames_dir=frames_dir)
    out_json = os.path.join(out_dir, "segments.json")
    write_json(out_json, {"video_id": vid, "video_path": video_path, "segments": segs})
    logging.info("Wrote %s (%d segments)", out_json, len(segs))
    return out_json


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--video", help="Path to one video file (.mp4)")
    p.add_argument("--raw_dir", default="datasets/raw_videos", help="Folder of raw videos")
    p.add_argument("--out_root", default="datasets/segments", help="Segments output root")
    p.add_argument("--segment_s", type=int, default=60)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)

    videos = [args.video] if args.video else sorted(glob(os.path.join(args.raw_dir, "*.mp4")))
    if not videos:
        raise FileNotFoundError("No videos found. Provide --video or populate datasets/raw_videos.")
    for vp in videos:
        segment_one(vp, args.out_root, args.segment_s)


if __name__ == "__main__":
    main()


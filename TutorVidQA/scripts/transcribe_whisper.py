"""
Transcribe videos with local Whisper (base) and align to segments.

Writes:
- datasets/transcripts/<video_id>.json
"""

from __future__ import annotations

import argparse
import logging
import os

from src.common import read_json, setup_logging, write_json
from src.transcriber import transcribe_and_align


def transcribe_video(video_path: str, segments_json: str, out_dir: str) -> str:
    """Run Whisper on `video_path` and save aligned transcript JSON."""
    seg_doc = read_json(segments_json)
    aligned = transcribe_and_align(video_path, seg_doc["segments"], model_name="base")
    out_path = os.path.join(out_dir, f"{seg_doc['video_id']}.json")
    write_json(out_path, {"video_id": seg_doc["video_id"], "chunks": aligned})
    logging.info("Wrote %s (%d chunks)", out_path, len(aligned))
    return out_path


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to video (.mp4)")
    p.add_argument("--segments_json", help="Path to segments.json (defaults by video_id)")
    p.add_argument("--segments_root", default="datasets/segments")
    p.add_argument("--out_dir", default="datasets/transcripts")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)

    vid = os.path.splitext(os.path.basename(args.video))[0]
    segments_json = args.segments_json or os.path.join(args.segments_root, vid, "segments.json")
    os.makedirs(args.out_dir, exist_ok=True)
    transcribe_video(args.video, segments_json, args.out_dir)


if __name__ == "__main__":
    main()


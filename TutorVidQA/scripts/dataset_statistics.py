"""
Compute basic dataset statistics for EduVidQA.

Outputs (printed):
- total videos
- total segments
- total transcript chunks
- total QA pairs
- average video duration
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

from src.common import read_json, read_jsonl, setup_logging


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--videos_meta", default="datasets/metadata/videos.jsonl")
    p.add_argument("--segments_root", default="datasets/segments")
    p.add_argument("--transcripts_dir", default="datasets/transcripts")
    p.add_argument("--qa_jsonl", default="datasets/annotations/qa_pairs.jsonl")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)

    videos = read_jsonl(args.videos_meta) if os.path.exists(args.videos_meta) else []
    durations = [v.get("duration_s") for v in videos if isinstance(v.get("duration_s"), (int, float))]

    seg_files = glob(os.path.join(args.segments_root, "*", "segments.json"))
    total_segments = sum(len(read_json(fp).get("segments", [])) for fp in seg_files)

    tr_files = glob(os.path.join(args.transcripts_dir, "*.json"))
    total_chunks = sum(len(read_json(fp).get("chunks", [])) for fp in tr_files)

    total_qas = len(read_jsonl(args.qa_jsonl)) if os.path.exists(args.qa_jsonl) else 0
    avg_dur = (sum(durations) / len(durations)) if durations else 0.0

    print(
        json.dumps(
            {
                "total_videos": len(videos),
                "total_segments": total_segments,
                "total_transcript_chunks": total_chunks,
                "total_qa_pairs": total_qas,
                "avg_video_duration_s": round(avg_dur, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


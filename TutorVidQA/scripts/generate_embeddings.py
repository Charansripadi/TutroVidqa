"""
Generate CLIP embeddings for transcript text + keyframes.

Writes:
- datasets/embeddings/<video_id>/text.npy
- datasets/embeddings/<video_id>/image.npy
- datasets/embeddings/<video_id>/meta.json
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np

from src.common import ensure_dir, read_json, setup_logging, write_json
from src.embedder import embed_images, embed_texts


def generate(video_id: str, segments_json: str, transcript_json: str, out_root: str) -> str:
    """Generate embeddings for a video_id and save arrays + metadata."""
    seg_doc = read_json(segments_json)
    tr_doc = read_json(transcript_json)

    Xt, mt = embed_texts(tr_doc["chunks"])
    Xi, mi = embed_images(seg_doc["segments"])

    out_dir = ensure_dir(os.path.join(out_root, video_id))
    np.save(os.path.join(out_dir, "text.npy"), Xt)
    np.save(os.path.join(out_dir, "image.npy"), Xi)
    write_json(os.path.join(out_dir, "meta.json"), {"text": mt, "image": mi})
    logging.info("Wrote embeddings to %s", out_dir)
    return out_dir


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True, help="Video id (filename stem)")
    p.add_argument("--segments_root", default="datasets/segments")
    p.add_argument("--transcripts_dir", default="datasets/transcripts")
    p.add_argument("--out_root", default="datasets/embeddings")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)

    segments_json = os.path.join(args.segments_root, args.video_id, "segments.json")
    transcript_json = os.path.join(args.transcripts_dir, f"{args.video_id}.json")
    generate(args.video_id, segments_json, transcript_json, args.out_root)


if __name__ == "__main__":
    main()


"""
Interactive TutorVidQA proof-of-concept.

This uses the dataset prep artifacts (segments + aligned transcript) to build
a temporary CLIP+FAISS retriever for a single video.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from src.common import setup_logging
from src.embedder import build_faiss_ip_index, embed_images, embed_texts
from src.retriever import retrieve_topk
from src.transcriber import transcribe_and_align
from src.video_segmenter import segment_video


def run(video_path: str, question: str, segment_s: int = 60, top_k: int = 5) -> None:
    """Run a minimal end-to-end demo on one video + question."""
    segments = segment_video(video_path, segment_duration_s=segment_s)
    aligned = transcribe_and_align(video_path, segments, model_name="base")

    Xt, mt = embed_texts(aligned)
    Xi, mi = embed_images(segments)
    X = Xt if Xi.size == 0 else (Xi if Xt.size == 0 else np.vstack([Xt, Xi]))
    meta = mt + mi

    index = build_faiss_ip_index(X)
    top = retrieve_topk(question, index, meta, k=top_k)

    print("\nRetrieved segments:")
    for r in top:
        ts = f"{r['start']:.1f}s–{r['end']:.1f}s"
        snippet = (r.get("text", "") or "").strip().replace("\n", " ")[:140]
        print(f"- seg={r['segment_id']} score={r['score']:.3f} {ts} via {r.get('modality')}")
        if snippet:
            print(f"  text: {snippet}...")
        if r.get("frame_path"):
            print(f"  frame: {r['frame_path']}")

    ctx = "\n".join(f"[{r['start']:.1f}-{r['end']:.1f}s] {r.get('text','')}".strip() for r in top)
    print("\nAnswer (stub):")
    print(json.dumps({"question": question, "grounding": ctx}, ensure_ascii=False, indent=2))
    print("\nProposed answer: (TODO) synthesize from grounding above using a local LLM.")


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to lecture video (.mp4)")
    p.add_argument("--question", required=True, help="Question to retrieve for")
    p.add_argument("--segment_s", type=int, default=60, help="Segment size in seconds")
    p.add_argument("--top_k", type=int, default=5, help="How many segments to return")
    p.add_argument("--log_level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = p.parse_args()
    setup_logging(args.log_level)
    logging.info("Running TutorVidQA demo on %s", os.path.basename(args.video))
    run(args.video, args.question, segment_s=args.segment_s, top_k=args.top_k)


if __name__ == "__main__":
    main()


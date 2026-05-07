"""
Generate timestamp-grounded QA pairs from aligned transcripts (no paid APIs).

This is a research-oriented baseline generator:
- definition questions from "X is/means/defined as ..."
- temporal reasoning questions across adjacent segments (before/after)
- simple multi-turn pairs (follow-up prompting for examples)

Writes:
- datasets/annotations/qa_pairs.jsonl
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from glob import glob
from typing import Dict, Iterable, List

from src.common import append_jsonl, read_json, setup_logging


_DEF_RE = re.compile(r"\b([A-Z][A-Za-z0-9 /_-]{2,60})\s+(is|means|refers to)\s+", re.I)


def _sentences(text: str) -> List[str]:
    """Very small sentence splitter for transcripts."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def generate_from_chunks(video_id: str, chunks: List[Dict]) -> Iterable[Dict]:
    """
    Create QA pairs from aligned transcript chunks.

    Output format:
    {"question","answer","timestamp","segment_id","video_id","type","turn"}
    """
    for c in chunks:
        seg_id = int(c["segment_id"])
        ts = f"{float(c['start']):.1f}-{float(c['end']):.1f}s"
        sents = _sentences(c.get("text", ""))
        for s in sents[:3]:
            m = _DEF_RE.search(s)
            if not m:
                continue
            concept = m.group(1).strip().strip("-")
            yield {
                "video_id": video_id,
                "segment_id": seg_id,
                "timestamp": ts,
                "type": "definition",
                "turn": 1,
                "question": f"What is {concept}?",
                "answer": s,
            }
            # Follow-up turn: example request (multi-turn conversational QA).
            yield {
                "video_id": video_id,
                "segment_id": seg_id,
                "timestamp": ts,
                "type": "followup_example",
                "turn": 2,
                "question": f"Can you give an example of {concept}?",
                "answer": (sents[1] if len(sents) > 1 else c.get("text", ""))[:300],
            }

    # Temporal reasoning: link adjacent segments (before/after).
    for i in range(1, len(chunks)):
        cur, prev = chunks[i], chunks[i - 1]
        if not cur.get("text") or not prev.get("text"):
            continue
        ts = f"{float(cur['start']):.1f}-{float(cur['end']):.1f}s"
        yield {
            "video_id": video_id,
            "segment_id": int(cur["segment_id"]),
            "timestamp": ts,
            "type": "temporal_before",
            "turn": 1,
            "question": "What was explained immediately before this section?",
            "answer": prev["text"][:400],
        }


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--transcripts_dir", default="datasets/transcripts")
    p.add_argument("--out_jsonl", default="datasets/annotations/qa_pairs.jsonl")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output JSONL")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    if args.overwrite and os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)
    files = sorted(glob(os.path.join(args.transcripts_dir, "*.json")))
    if not files:
        raise FileNotFoundError("No transcripts found. Run scripts/transcribe_whisper.py first.")

    total = 0
    for fp in files:
        doc = read_json(fp)
        rows = list(generate_from_chunks(doc["video_id"], doc["chunks"]))
        append_jsonl(args.out_jsonl, rows)
        total += len(rows)
        logging.info("Generated %d QA pairs for %s", len(rows), doc["video_id"])
    logging.info("Total QA pairs appended: %d", total)


if __name__ == "__main__":
    main()


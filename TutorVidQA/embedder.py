from __future__ import annotations

from typing import Dict, List, Tuple, Any

import faiss
import numpy as np

from src.embedder import embed_texts


def create_embeddings(transcripts: List[Dict], segments: List[Dict]) -> Tuple[Any, List[Dict]]:
    """
    Creates CLIP embeddings and stores them in FAISS.

    This PoC version builds a text-only index (compatible with your snippet),
    while reusing the robust CLIP encoder in `src.embedder`.
    """
    X, meta = embed_texts(transcripts)
    # Use L2 (as in the snippet) over normalized vectors. This is fine for a PoC.
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(np.asarray(X, dtype="float32"))

    # Match the snippet's metadata shape (segment-level text).
    by_segment = {int(m["segment_id"]): m for m in meta}
    out_meta: List[Dict] = []
    for seg in segments:
        sid = int(seg["segment_id"])
        m = by_segment.get(sid, {})
        out_meta.append(
            {
                "segment_id": sid,
                "start": float(m.get("start", seg.get("start_time", 0.0))),
                "end": float(m.get("end", seg.get("end_time", 0.0))),
                "text": m.get("text", ""),
            }
        )

    return index, out_meta


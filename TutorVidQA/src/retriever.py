"""
Retrieval over FAISS using CLIP query embeddings.

This is used both for interactive QA and for analysis/benchmarking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize 2D array rows."""
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def encode_query(
    query: str,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
) -> np.ndarray:
    """Encode a query string into a normalized CLIP text embedding (shape [1,D])."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    proc = CLIPProcessor.from_pretrained(clip_model_name)
    inputs = proc(text=[query], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        q = model.get_text_features(**inputs).cpu().numpy().astype("float32")
    return _l2_normalize(q)


def retrieve_topk(
    query: str,
    index: Any,
    metadata: List[Dict],
    k: int = 5,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve top-k most relevant segments for `query`.

    Returns a segment-level ranking (aggregating multiple modalities per segment).
    """
    q = encode_query(query, clip_model_name=clip_model_name, device=device)
    scores, idxs = index.search(q, min(k * 5, len(metadata)))
    hits = [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    best: Dict[int, Dict] = {}
    for i, score in hits:
        m = metadata[i]
        sid = int(m["segment_id"])
        if sid not in best or score > best[sid]["score"]:
            best[sid] = {"segment_id": sid, "score": score, **m}
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:k]


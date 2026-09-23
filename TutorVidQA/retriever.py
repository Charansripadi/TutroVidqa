from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.retriever import encode_query


def retrieve_segments(query: str, index: Any, metadata: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Retrieves most relevant segments using CLIP embeddings.

    PoC wrapper: uses the same CLIP query encoder as `src.retriever`, but returns
    the simple metadata objects like your snippet.
    """
    q = encode_query(query)
    distances, indices = index.search(np.asarray(q, dtype="float32"), top_k)
    results: List[Dict] = []
    for idx in indices[0]:
        if int(idx) < 0:
            continue
        results.append(metadata[int(idx)])
    return results


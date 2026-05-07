"""
Build and persist a FAISS index from saved embeddings.

Writes:
- datasets/embeddings/faiss.index
- datasets/embeddings/faiss_meta.json
"""

from __future__ import annotations

import argparse
import logging
import os
from glob import glob

import faiss
import numpy as np

from src.common import read_json, setup_logging, write_json
from src.embedder import build_faiss_ip_index


def build(emb_root: str, out_index: str, out_meta: str) -> None:
    """Load all per-video embeddings, build a single FAISS index, and save it."""
    metas, vecs = [], []
    for meta_path in sorted(glob(os.path.join(emb_root, "*", "meta.json"))):
        video_dir = os.path.dirname(meta_path)
        mt = read_json(meta_path)
        Xt = np.load(os.path.join(video_dir, "text.npy"))
        Xi = np.load(os.path.join(video_dir, "image.npy"))
        vecs.append(np.vstack([Xt, Xi]))
        metas.extend(mt["text"] + mt["image"])

    if not vecs:
        raise FileNotFoundError("No embeddings found. Run scripts/generate_embeddings.py first.")
    X = np.vstack(vecs).astype("float32")
    index = build_faiss_ip_index(X)
    faiss.write_index(index, out_index)
    write_json(out_meta, {"count": len(metas), "metadata": metas})
    logging.info("Saved FAISS index=%s vectors=%d", out_index, X.shape[0])


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--emb_root", default="datasets/embeddings")
    p.add_argument("--out_index", default="datasets/embeddings/faiss.index")
    p.add_argument("--out_meta", default="datasets/embeddings/faiss_meta.json")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)

    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    build(args.emb_root, args.out_index, args.out_meta)


if __name__ == "__main__":
    main()


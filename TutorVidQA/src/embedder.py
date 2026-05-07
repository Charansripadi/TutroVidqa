"""
CLIP embedding generation for EduVidQA.

Generates text and image embeddings with a local CLIP model (HuggingFace).
Embeddings can be saved as NumPy arrays and/or indexed in FAISS.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows for cosine similarity with inner product."""
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def embed_texts(
    text_chunks: List[Dict],
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Embed a list of {segment_id,start,end,text} using CLIP text encoder.

    Returns (embeddings[N,D], metadata[N]) with modality="text".
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    proc = CLIPProcessor.from_pretrained(clip_model_name)

    texts = [c.get("text", "") for c in text_chunks]
    inputs = proc(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        X = model.get_text_features(**inputs).cpu().numpy().astype("float32")
    X = _l2_normalize(X)

    meta = [
        {
            "segment_id": int(c["segment_id"]),
            "modality": "text",
            "start": float(c["start"]),
            "end": float(c["end"]),
            "text": c.get("text", ""),
        }
        for c in text_chunks
    ]
    logging.info("Embedded %d text chunks dim=%d", X.shape[0], X.shape[1])
    return X, meta


def embed_images(
    frame_segments: List[Dict],
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Embed frames for each segment using CLIP image encoder.

    Returns (embeddings[N,D], metadata[N]) with modality="image".
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    proc = CLIPProcessor.from_pretrained(clip_model_name)

    images = [Image.open(s["frame_path"]).convert("RGB") for s in frame_segments]
    inputs = proc(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        X = model.get_image_features(**inputs).cpu().numpy().astype("float32")
    X = _l2_normalize(X)

    meta = [
        {
            "segment_id": int(s["segment_id"]),
            "modality": "image",
            "start": float(s["start_time"]),
            "end": float(s["end_time"]),
            "frame_path": s["frame_path"],
        }
        for s in frame_segments
    ]
    logging.info("Embedded %d images dim=%d", X.shape[0], X.shape[1])
    return X, meta


def build_faiss_ip_index(vectors: np.ndarray) -> Any:
    """
    Build a FAISS cosine-similarity index using inner product on normalized vectors.
    """
    vectors = vectors.astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


"""
Whisper transcription + alignment for EduVidQA.

We transcribe audio once with local Whisper, then align Whisper's timestamped
segments to fixed video segments using temporal overlap.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import whisper


def transcribe_and_align(
    media_path: str, video_segments: List[Dict], model_name: str = "base"
) -> List[Dict]:
    """
    Transcribe `media_path` using local Whisper and align to `video_segments`.

    Returns: list of {segment_id, start, end, text}
    Key steps:
    - run Whisper to get timestamped chunks
    - for each segment window, merge overlapping Whisper text
    """
    logging.info("Loading Whisper model=%s", model_name)
    model = whisper.load_model(model_name)
    logging.info("Transcribing %s", media_path)
    result = model.transcribe(media_path, verbose=False)
    wsegs = result.get("segments", [])

    aligned: List[Dict] = []
    for seg in video_segments:
        s0, s1 = float(seg["start_time"]), float(seg["end_time"])
        texts = [
            ws.get("text", "").strip()
            for ws in wsegs
            if float(ws.get("end", 0.0)) > s0 and float(ws.get("start", 0.0)) < s1
        ]
        aligned.append(
            {
                "segment_id": int(seg["segment_id"]),
                "start": s0,
                "end": s1,
                "text": " ".join(t for t in texts if t),
            }
        )

    logging.info("Aligned transcript into %d segment chunks", len(aligned))
    return aligned


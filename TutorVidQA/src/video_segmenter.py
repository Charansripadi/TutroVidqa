"""
Video segmentation for EduVidQA.

Splits a video into fixed-duration segments and saves one representative frame
per segment. The resulting JSON is the canonical segment map used by the rest
of the dataset pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import cv2


@dataclass(frozen=True)
class VideoSegment:
    """Metadata for one video segment."""

    segment_id: int
    start_time: float
    end_time: float
    frame_path: str


def segment_video(
    video_path: str,
    segment_duration_s: int = 60,
    frames_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Split `video_path` into segments of ~`segment_duration_s` seconds.

    Key steps:
    - open video with OpenCV
    - seek to each segment start time, save a keyframe image
    - return list of {segment_id,start_time,end_time,frame_path}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (total_frames / fps) if total_frames else 0.0

    frames_dir = frames_dir or os.path.join(os.path.dirname(video_path), "keyframes")
    os.makedirs(frames_dir, exist_ok=True)
    logging.info("Segmenting video fps=%.2f duration=%.1fs", fps, duration_s)

    segments: List[VideoSegment] = []
    seg_id, t = 0, 0.0
    while duration_s == 0.0 or t < duration_s:
        start_t, end_t = t, t + float(segment_duration_s)
        start_frame = int(round(start_t * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = cap.read()
        if not ok:
            break

        frame_path = os.path.join(frames_dir, f"segment_{seg_id:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        segments.append(VideoSegment(seg_id, start_t, end_t, frame_path))
        seg_id, t = seg_id + 1, end_t

    cap.release()
    logging.info("Created %d segments", len(segments))
    return [asdict(s) for s in segments]


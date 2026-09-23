import os
from typing import List, Dict

from src.video_segmenter import segment_video as _segment_video


def segment_video(video_path: str, segment_duration: int = 60, output_dir: str = "frames") -> List[Dict]:
    """
    Splits a video into time-based segments and extracts one frame.

    This is a PoC-friendly wrapper around `src.video_segmenter.segment_video`.
    """
    os.makedirs(output_dir, exist_ok=True)
    return _segment_video(
        video_path=video_path,
        segment_duration_s=segment_duration,
        frames_dir=output_dir,
    )


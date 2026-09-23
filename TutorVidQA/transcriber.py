from typing import List, Dict

from src.transcriber import transcribe_and_align as _transcribe_and_align


def transcribe_video(video_path: str, segments: List[Dict]) -> List[Dict]:
    """
    Transcribes video using Whisper and aligns text to segments.

    PoC wrapper around `src.transcriber.transcribe_and_align`.
    """
    return _transcribe_and_align(video_path, segments, model_name="base")


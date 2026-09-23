from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Box:
    label: str
    xy: Tuple[int, int]
    wh: Tuple[int, int]
    fc: str = "#ffffff"
    ec: str = "#111827"  # gray-900


def _load_font(size: int) -> ImageFont.ImageFont:
    # Pillow fallback font (portable in sandbox).
    return ImageFont.load_default(size=size)


def _rounded_rect(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, r: int, fill: str, outline: str):
    draw.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=fill, outline=outline, width=2)


def _draw_centered_text(draw: ImageDraw.ImageDraw, box: Box, text: str, font: ImageFont.ImageFont, fill: str) -> None:
    x, y = box.xy
    w, h = box.wh
    lines = text.split("\n")
    # simple line height approximation
    line_h = int(font.size * 1.25)
    total_h = line_h * len(lines)
    y0 = y + (h - total_h) // 2
    for i, line in enumerate(lines):
        tw = draw.textlength(line, font=font)
        tx = x + (w - int(tw)) // 2
        ty = y0 + i * line_h
        draw.text((tx, ty), line, font=font, fill=fill)


def _add_box(draw: ImageDraw.ImageDraw, b: Box, font: ImageFont.ImageFont) -> None:
    x, y = b.xy
    w, h = b.wh
    _rounded_rect(draw, x, y, x + w, y + h, r=16, fill=b.fc, outline=b.ec)
    _draw_centered_text(draw, b, b.label, font=font, fill="#0f172a")


def _arrow(draw: ImageDraw.ImageDraw, a: Box, b: Box, label: str | None, font: ImageFont.ImageFont, color: str = "#0f172a") -> None:
    ax = a.xy[0] + a.wh[0]
    ay = a.xy[1] + a.wh[1] // 2
    bx = b.xy[0]
    by = b.xy[1] + b.wh[1] // 2
    draw.line((ax, ay, bx, by), fill=color, width=3)
    # arrow head
    head = 10
    draw.polygon([(bx, by), (bx - head, by - head // 2), (bx - head, by + head // 2)], fill=color)
    if label:
        mx = (ax + bx) // 2
        my = ay - 18
        draw.text((mx - int(draw.textlength(label, font=font)) // 2, my), label, font=font, fill="#334155")


def render_overview(out_path: str) -> None:
    img = Image.new("RGB", (1800, 900), "white")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(22)
    font = _load_font(14)
    small = _load_font(12)

    title = "TutorVidQA: Proof-of-Concept Pipeline"
    draw.text((900 - int(draw.textlength(title, font=title_font)) // 2, 30), title, font=title_font, fill="#0f172a")

    navy = "#1e3a8a"
    teal = "#0f766e"

    boxes: List[Box] = [
        Box("Educational Video\nInput (.mp4)", (60, 600), (280, 150)),
        Box("Video Segmenter", (400, 620), (240, 110), fc="#f8fafc"),
        Box("Keyframe\nExtraction", (720, 540), (250, 110), fc="#ecfeff", ec=teal),
        Box("Whisper ASR", (720, 700), (250, 110), fc="#eff6ff", ec=navy),
        Box("CLIP Embedder\n(text + image)", (1040, 540), (270, 110), fc="#ecfeff", ec=teal),
        Box("Transcript\nAlignment", (1040, 700), (270, 110), fc="#eff6ff", ec=navy),
        Box("FAISS Vector\nStore", (1380, 620), (260, 140), fc="#f8fafc"),
        Box("User Question", (60, 280), (280, 120)),
        Box("Query Encoder\n(CLIP text)", (400, 285), (240, 110), fc="#ecfeff", ec=teal),
        Box("Retriever", (720, 285), (250, 110), fc="#f8fafc"),
        Box("Grounded QA\nModule", (1040, 285), (270, 110), fc="#f8fafc"),
        Box("Timestamp-grounded\nResponse", (1380, 270), (260, 140)),
    ]

    by_label = {b.label: b for b in boxes}
    for b in boxes:
        _add_box(draw, b, font=font)

    _arrow(draw, by_label["Educational Video\nInput (.mp4)"], by_label["Video Segmenter"], "segment", small)
    _arrow(draw, by_label["Video Segmenter"], by_label["Keyframe\nExtraction"], "extract frame", small, color=teal)
    _arrow(draw, by_label["Video Segmenter"], by_label["Whisper ASR"], "audio → text", small, color=navy)
    _arrow(draw, by_label["Keyframe\nExtraction"], by_label["CLIP Embedder\n(text + image)"], "embed image", small, color=teal)
    _arrow(draw, by_label["Whisper ASR"], by_label["Transcript\nAlignment"], "align", small, color=navy)
    _arrow(draw, by_label["Transcript\nAlignment"], by_label["CLIP Embedder\n(text + image)"], "embed text", small, color=teal)
    _arrow(draw, by_label["CLIP Embedder\n(text + image)"], by_label["FAISS Vector\nStore"], "index", small)

    _arrow(draw, by_label["User Question"], by_label["Query Encoder\n(CLIP text)"], "encode", small, color=teal)
    _arrow(draw, by_label["Query Encoder\n(CLIP text)"], by_label["Retriever"], "search", small)
    _arrow(draw, by_label["Retriever"], by_label["Grounded QA\nModule"], "evidence", small)
    _arrow(draw, by_label["Grounded QA\nModule"], by_label["Timestamp-grounded\nResponse"], "answer + citations", small)

    # Arrow from FAISS down to Retriever
    f = by_label["FAISS Vector\nStore"]
    r = by_label["Retriever"]
    fx = f.xy[0] + f.wh[0] // 2
    fy = f.xy[1]
    rx = r.xy[0] + r.wh[0] // 2
    ry = r.xy[1] + r.wh[1]
    draw.line((fx, fy, rx, ry), fill="#0f172a", width=3)
    draw.polygon([(rx, ry), (rx - 8, ry + 10), (rx + 8, ry + 10)], fill="#0f172a")
    draw.text((900 - int(draw.textlength("retrieve top‑K", font=small)) // 2, 500), "retrieve top‑K", font=small, fill="#334155")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format="PNG")


def render_retrieval_flow(out_path: str) -> None:
    img = Image.new("RGB", (1600, 500), "white")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(20)
    font = _load_font(14)
    small = _load_font(12)

    teal = "#0f766e"
    boxes: List[Box] = [
        Box("User Question", (60, 220), (240, 120)),
        Box("Query\nEmbedding", (340, 220), (240, 120), fc="#ecfeff", ec=teal),
        Box("Similarity\nSearch (FAISS)", (620, 220), (300, 120), fc="#f8fafc"),
        Box("Top‑K Segments\n(+ timestamps)", (960, 220), (320, 120), fc="#f8fafc"),
        Box("Evidence\nContext", (1320, 220), (240, 120), fc="#f8fafc"),
    ]
    for b in boxes:
        _add_box(draw, b, font=font)
    for i in range(len(boxes) - 1):
        _arrow(draw, boxes[i], boxes[i + 1], None, small, color="#0f172a")

    title = "Retrieval-Augmented QA (timestamp-grounded)"
    draw.text((800 - int(draw.textlength(title, font=title_font)) // 2, 40), title, font=title_font, fill="#0f172a")

    out_box = Box("Grounded Answer\n(with citations)", (1320, 70), (240, 110), fc="#ecfeff", ec=teal)
    _add_box(draw, out_box, font=font)
    # Arrow up from evidence to answer
    ex = boxes[-1].xy[0] + boxes[-1].wh[0] // 2
    ey = boxes[-1].xy[1]
    ox = out_box.xy[0] + out_box.wh[0] // 2
    oy = out_box.xy[1] + out_box.wh[1]
    draw.line((ex, ey, ox, oy), fill="#0f172a", width=3)
    draw.polygon([(ox, oy), (ox - 8, oy + 10), (ox + 8, oy + 10)], fill="#0f172a")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    render_overview("docs/architecture/architecture_overview.png")
    render_retrieval_flow("docs/architecture/retrieval_flow.png")
    print("Wrote docs/architecture/*.png")


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


@dataclass(frozen=True)
class Box:
    label: str
    xy: Tuple[int, int]
    wh: Tuple[int, int]
    fc: str = "#ffffff"
    ec: str = "#111827"


def _box(draw: ImageDraw.ImageDraw, b: Box, font: ImageFont.ImageFont) -> None:
    x, y = b.xy
    w, h = b.wh
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=b.fc, outline=b.ec, width=2)
    # centered multiline
    lines = b.label.split("\n")
    line_h = int(font.size * 1.25)
    total_h = line_h * len(lines)
    y0 = y + (h - total_h) // 2
    for i, line in enumerate(lines):
        tw = int(draw.textlength(line, font=font))
        tx = x + (w - tw) // 2
        ty = y0 + i * line_h
        draw.text((tx, ty), line, font=font, fill="#0f172a")


def _arrow(draw: ImageDraw.ImageDraw, a: Box, b: Box, label: str | None, font: ImageFont.ImageFont, color: str = "#0f172a") -> None:
    ax = a.xy[0] + a.wh[0]
    ay = a.xy[1] + a.wh[1] // 2
    bx = b.xy[0]
    by = b.xy[1] + b.wh[1] // 2
    draw.line((ax, ay, bx, by), fill=color, width=3)
    head = 10
    draw.polygon([(bx, by), (bx - head, by - head // 2), (bx - head, by + head // 2)], fill=color)
    if label:
        mx = (ax + bx) // 2
        my = ay - 18
        draw.text((mx - int(draw.textlength(label, font=font)) // 2, my), label, font=font, fill="#334155")


def render_dataset_artifact_flow(out_path: str) -> None:
    W, H = 1800, 850
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(22)
    font = _font(14)
    small = _font(12)

    title = "EduVidQA dataset pipeline (artifacts + scripts)"
    d.text((W // 2 - int(d.textlength(title, font=title_font)) // 2, 25), title, font=title_font, fill="#0f172a")

    teal = "#0f766e"
    navy = "#1e3a8a"

    boxes: List[Box] = [
        Box("YouTube URL", (60, 130), (240, 110), fc="#f8fafc"),
        Box("scripts/download_youtube.py", (350, 130), (320, 110), fc="#ecfeff", ec=teal),
        Box("datasets/raw_videos/\n<video_id>.mp4", (720, 120), (300, 130), fc="#f8fafc"),
        Box("scripts/segment_videos.py", (1060, 70), (300, 110), fc="#ecfeff", ec=teal),
        Box("datasets/segments/<video_id>/\nsegments.json", (1400, 40), (330, 120), fc="#f8fafc"),
        Box("datasets/segments/<video_id>/\nkeyframes/*.jpg", (1400, 190), (330, 120), fc="#f8fafc"),
        Box("scripts/transcribe_whisper.py", (1060, 250), (300, 110), fc="#eff6ff", ec=navy),
        Box("datasets/transcripts/\n<video_id>.json", (720, 250), (300, 130), fc="#f8fafc"),
        Box("scripts/generate_embeddings.py", (350, 420), (320, 110), fc="#ecfeff", ec=teal),
        Box("datasets/embeddings/<video_id>/\ntext.npy + image.npy", (720, 420), (300, 130), fc="#f8fafc"),
        Box("scripts/build_faiss_index.py", (1060, 460), (300, 110), fc="#ecfeff", ec=teal),
        Box("datasets/embeddings/\nfaiss.index", (1400, 460), (330, 120), fc="#f8fafc"),
        Box("scripts/create_qa_pairs.py", (350, 610), (320, 110), fc="#f8fafc"),
        Box("datasets/annotations/\nqa_pairs.jsonl", (720, 610), (300, 130), fc="#f8fafc"),
        Box("src/qa_pipeline.py\n(demo)", (1060, 650), (300, 110), fc="#f8fafc"),
    ]
    by = {b.label: b for b in boxes}
    for b in boxes:
        _box(d, b, font=font)

    _arrow(d, by["YouTube URL"], by["scripts/download_youtube.py"], None, small, color=teal)
    _arrow(d, by["scripts/download_youtube.py"], by["datasets/raw_videos/\n<video_id>.mp4"], "download", small, color=teal)
    _arrow(d, by["datasets/raw_videos/\n<video_id>.mp4"], by["scripts/segment_videos.py"], None, small, color=teal)
    _arrow(d, by["scripts/segment_videos.py"], by["datasets/segments/<video_id>/\nsegments.json"], "segment map", small, color=teal)
    _arrow(d, by["scripts/segment_videos.py"], by["datasets/segments/<video_id>/\nkeyframes/*.jpg"], "keyframes", small, color=teal)
    _arrow(d, by["datasets/raw_videos/\n<video_id>.mp4"], by["scripts/transcribe_whisper.py"], None, small, color=navy)
    _arrow(d, by["scripts/transcribe_whisper.py"], by["datasets/transcripts/\n<video_id>.json"], "ASR", small, color=navy)
    _arrow(d, by["datasets/transcripts/\n<video_id>.json"], by["scripts/generate_embeddings.py"], None, small, color=teal)
    _arrow(d, by["datasets/segments/<video_id>/\nkeyframes/*.jpg"], by["scripts/generate_embeddings.py"], None, small, color=teal)
    _arrow(d, by["scripts/generate_embeddings.py"], by["datasets/embeddings/<video_id>/\ntext.npy + image.npy"], "CLIP", small, color=teal)
    _arrow(d, by["datasets/embeddings/<video_id>/\ntext.npy + image.npy"], by["scripts/build_faiss_index.py"], None, small, color=teal)
    _arrow(d, by["scripts/build_faiss_index.py"], by["datasets/embeddings/\nfaiss.index"], "index", small, color=teal)
    _arrow(d, by["datasets/embeddings/\nfaiss.index"], by["src/qa_pipeline.py\n(demo)"], "retrieve", small, color="#0f172a")
    _arrow(d, by["datasets/transcripts/\n<video_id>.json"], by["scripts/create_qa_pairs.py"], None, small, color="#0f172a")
    _arrow(d, by["scripts/create_qa_pairs.py"], by["datasets/annotations/\nqa_pairs.jsonl"], "generate", small, color="#0f172a")

    d.text((60, H - 55), "Exported PNG (regenerate via scripts/render_more_report_images.py).", font=small, fill="#475569")
    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="PNG")


def render_pr_curves(pr_curve: Dict[str, List[float]], out_path: str) -> None:
    # K is 1..len(list)
    ks = list(range(1, len(pr_curve["precision"]) + 1))
    prec = pr_curve["precision"]
    rec = pr_curve["recall"]

    W, H = 1200, 700
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(20)
    small = _font(12)
    label = _font(14)

    title = "Precision@K and Recall@K (curve)"
    d.text((W // 2 - int(d.textlength(title, font=title_font)) // 2, 20), title, font=title_font, fill="#0f172a")

    x0, y0 = 90, 90
    x1, y1 = W - 50, H - 120
    d.line((x0, y1, x1, y1), fill="#111827", width=2)
    d.line((x0, y0, x0, y1), fill="#111827", width=2)
    d.text((20, (y0 + y1) // 2), "Score", font=label, fill="#334155")
    d.text((W // 2, H - 90), "K", font=label, fill="#334155")

    # grid
    for i in range(0, 11, 2):
        y = y1 - int((y1 - y0) * (i / 10.0))
        d.line((x0, y, x1, y), fill="#e2e8f0", width=1)
        d.text((x0 - 35, y - 6), f"{i/10:.1f}", font=small, fill="#64748b")

    def pt(k: int, v: float) -> Tuple[int, int]:
        xx = x0 + int((x1 - x0) * ((k - 1) / max(1, (len(ks) - 1))))
        yy = y1 - int((y1 - y0) * max(0.0, min(1.0, v)))
        return xx, yy

    # ticks on x
    for k in ks:
        if k == 1 or k == len(ks) or k % 2 == 1:
            x, _ = pt(k, 0.0)
            d.line((x, y1, x, y1 + 6), fill="#111827", width=1)
            d.text((x - 4, y1 + 10), str(k), font=small, fill="#64748b")

    # curves
    def draw_curve(values: List[float], color: str, name: str) -> None:
        pts = [pt(k, v) for k, v in zip(ks, values)]
        for a, b in zip(pts, pts[1:]):
            d.line((*a, *b), fill=color, width=3)
        for x, y in pts:
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=color)
        d.text((x1 - 220, y0 + (20 if name == "Precision" else 42)), f"{name}", font=label, fill=color)

    draw_curve(prec, "#1e3a8a", "Precision")
    draw_curve(rec, "#0f766e", "Recall")

    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="PNG")


def render_latency_breakdown(latencies: Dict[str, float], out_path: str) -> None:
    # simple stacked horizontal bar
    stages = list(latencies.keys())
    vals = [float(latencies[s]) for s in stages]
    total = sum(vals) or 1.0

    W, H = 1200, 520
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(20)
    label = _font(14)
    small = _font(12)

    title = "Latency breakdown (end-to-end)"
    d.text((W // 2 - int(d.textlength(title, font=title_font)) // 2, 20), title, font=title_font, fill="#0f172a")

    x0, y0 = 100, 150
    bar_w, bar_h = 980, 44
    d.rounded_rectangle((x0, y0, x0 + bar_w, y0 + bar_h), radius=18, fill="#f1f5f9", outline="#cbd5e1", width=1)

    palette = ["#1e3a8a", "#0f766e", "#7c3aed", "#ea580c", "#475569"]
    cur = x0
    for i, (stage, v) in enumerate(zip(stages, vals)):
        w = int(bar_w * (v / total))
        if i == len(stages) - 1:
            w = x0 + bar_w - cur
        d.rounded_rectangle((cur, y0, cur + w, y0 + bar_h), radius=18, fill=palette[i % len(palette)])
        d.text((cur + 8, y0 + bar_h + 10), f"{stage}: {v:.2f}s", font=small, fill="#334155")
        cur += w

    d.text((x0, 110), f"Total: {total:.2f}s", font=label, fill="#0f172a")
    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="PNG")


def render_ablation(ablation: Dict[str, float], out_path: str) -> None:
    # grouped bars for 3-4 variants
    names = list(ablation.keys())
    vals = [float(ablation[n]) for n in names]

    W, H = 1200, 650
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(20)
    label = _font(14)
    small = _font(12)

    title = "Ablation (example): retrieval setting vs score"
    d.text((W // 2 - int(d.textlength(title, font=title_font)) // 2, 20), title, font=title_font, fill="#0f172a")

    x0, y0 = 90, 90
    x1, y1 = W - 60, H - 140
    d.line((x0, y1, x1, y1), fill="#111827", width=2)
    d.line((x0, y0, x0, y1), fill="#111827", width=2)
    d.text((20, (y0 + y1) // 2), "Score (0–1)", font=label, fill="#334155")

    # grid
    for i in range(0, 11, 2):
        y = y1 - int((y1 - y0) * (i / 10.0))
        d.line((x0, y, x1, y), fill="#e2e8f0", width=1)
        d.text((x0 - 35, y - 6), f"{i/10:.1f}", font=small, fill="#64748b")

    n = len(names)
    gap = 24
    bar_w = int((x1 - x0 - gap * (n + 1)) / max(1, n))
    colors = ["#1e3a8a", "#0f766e", "#7c3aed", "#ea580c"]
    for i, (name, v) in enumerate(zip(names, vals)):
        bx0 = x0 + gap + i * (bar_w + gap)
        bx1 = bx0 + bar_w
        by1 = y1
        by0 = y1 - int((y1 - y0) * max(0.0, min(1.0, v)))
        d.rectangle((bx0, by0, bx1, by1), fill=colors[i % len(colors)], outline="#0f172a", width=1)
        d.text((bx0 + 4, by0 - 18), f"{v:.2f}", font=small, fill="#0f172a")
        d.text((bx0, y1 + 12), name, font=small, fill="#334155")

    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="PNG")


def render_qualitative_grid(examples: List[Dict[str, str]], out_path: str) -> None:
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(20)
    small = _font(12)
    label = _font(14)

    title = "Qualitative examples (template)"
    d.text((W // 2 - int(d.textlength(title, font=title_font)) // 2, 20), title, font=title_font, fill="#0f172a")

    cols = 2
    rows = 2
    card_w = (W - 120 - 40) // cols
    card_h = (H - 120 - 40) // rows
    x0, y0 = 60, 80
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(examples):
                break
            ex = examples[idx]
            cx = x0 + c * (card_w + 40)
            cy = y0 + r * (card_h + 40)
            d.rounded_rectangle((cx, cy, cx + card_w, cy + card_h), radius=18, fill="#f8fafc", outline="#cbd5e1", width=2)
            d.text((cx + 16, cy + 14), f"Q{idx+1}: {ex.get('question','')}", font=label, fill="#0f172a")
            d.text((cx + 16, cy + 60), f"GT: {ex.get('gt','')}", font=small, fill="#334155")
            d.text((cx + 16, cy + 84), f"Retrieved: {ex.get('retrieved','')}", font=small, fill="#334155")
            d.text((cx + 16, cy + 120), "Evidence snippet:", font=small, fill="#334155")
            snippet = (ex.get("snippet", "") or "")[:260]
            # naive wrap
            line_len = 56
            lines = [snippet[i : i + line_len] for i in range(0, len(snippet), line_len)]
            for i, line in enumerate(lines[:6]):
                d.text((cx + 16, cy + 142 + i * 18), line, font=small, fill="#475569")
            idx += 1

    d.text((60, H - 45), "Fill `docs/evaluation/example_qualitative.json` and rerun to render real examples.", font=small, fill="#475569")
    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="PNG")


def main() -> None:
    _ensure_dir("docs/architecture")
    _ensure_dir("docs/evaluation")

    render_dataset_artifact_flow("docs/architecture/dataset_artifact_flow.png")

    pr_curve = {
        "precision": [0.78, 0.82, 0.85, 0.86, 0.87, 0.87, 0.88, 0.88, 0.88, 0.89],
        "recall": [0.45, 0.60, 0.72, 0.78, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89],
    }
    render_pr_curves(pr_curve, "docs/evaluation/pr_curves.png")

    latencies = {"segment": 0.20, "whisper": 8.30, "embed": 1.10, "index": 0.15, "retrieve": 0.08}
    render_latency_breakdown(latencies, "docs/evaluation/latency_breakdown.png")

    ablation = {"text-only": 0.84, "image-only": 0.61, "text+image": 0.87}
    render_ablation(ablation, "docs/evaluation/ablation.png")

    examples = [
        {
            "question": "What is Retrieval-Augmented Generation?",
            "gt": "120s–180s (seg 2)",
            "retrieved": "118s–178s (seg 2)",
            "snippet": "Retrieval-Augmented Generation combines retrieval systems with language models...",
        },
        {
            "question": "When does the instructor introduce vector databases?",
            "gt": "300s–360s (seg 5)",
            "retrieved": "240s–300s (seg 4) (miss)",
            "snippet": "We will store embeddings in FAISS and run similarity search to fetch relevant context...",
        },
        {"question": "Define precision and recall.", "gt": "—", "retrieved": "—", "snippet": "—"},
        {"question": "Explain CLIP embeddings.", "gt": "—", "retrieved": "—", "snippet": "—"},
    ]
    render_qualitative_grid(examples, "docs/evaluation/qualitative_examples.png")

    # export example inputs for editing
    with open("docs/evaluation/example_more_inputs.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "pr_curve": pr_curve,
                "latency_breakdown": latencies,
                "ablation": ablation,
                "qualitative_examples": examples,
            },
            f,
            indent=2,
        )

    print("Wrote docs/architecture/dataset_artifact_flow.png and additional docs/evaluation/*.png")


if __name__ == "__main__":
    main()


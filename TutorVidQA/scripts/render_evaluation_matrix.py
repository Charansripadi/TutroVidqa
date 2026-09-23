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
class Cell:
    label: str
    value: float
    xy: Tuple[int, int]
    wh: Tuple[int, int]


def render_confusion_matrix(tp: int, fp: int, fn: int, tn: int, out_path: str) -> None:
    """
    Render a simple 2x2 confusion-matrix style figure.

    In this project it can represent retrieval at a chosen K:
    - Positive = "relevant segment retrieved"
    - Negative = "irrelevant segment retrieved / missed"
    """
    W, H = 1100, 700
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    title_font = _font(20)
    label_font = _font(14)
    big = _font(18)
    small = _font(12)

    title = "Evaluation matrix (confusion-matrix style)"
    d.text((W // 2 - int(d.textlength(title, font=title_font)) // 2, 20), title, font=title_font, fill="#0f172a")

    # axes labels
    d.text((120, 90), "Ground truth", font=label_font, fill="#334155")
    d.text((700, 90), "Predicted", font=label_font, fill="#334155")

    grid_x0, grid_y0 = 120, 140
    cell_w, cell_h = 360, 220
    gap = 20

    cells: List[Cell] = [
        Cell("TP\n(relevant retrieved)", float(tp), (grid_x0, grid_y0), (cell_w, cell_h)),
        Cell("FP\n(irrelevant retrieved)", float(fp), (grid_x0 + cell_w + gap, grid_y0), (cell_w, cell_h)),
        Cell("FN\n(relevant missed)", float(fn), (grid_x0, grid_y0 + cell_h + gap), (cell_w, cell_h)),
        Cell("TN\n(irrelevant not retrieved)", float(tn), (grid_x0 + cell_w + gap, grid_y0 + cell_h + gap), (cell_w, cell_h)),
    ]

    # color intensity by value (simple normalization)
    vmax = max(1.0, float(max(tp, fp, fn, tn)))

    def shade(v: float) -> str:
        # more blue as value increases
        t = min(1.0, max(0.0, v / vmax))
        base = (248, 250, 252)  # slate-50
        hi = (30, 58, 138)  # blue-900
        r = int(base[0] * (1 - t) + hi[0] * t)
        g = int(base[1] * (1 - t) + hi[1] * t)
        b = int(base[2] * (1 - t) + hi[2] * t)
        return (r, g, b)

    for c in cells:
        x, y = c.xy
        w, h = c.wh
        d.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=shade(c.value), outline="#111827", width=2)
        # labels (top-left)
        d.text((x + 16, y + 12), c.label, font=label_font, fill="white" if c.value / vmax > 0.35 else "#0f172a")
        # value (center)
        val = f"{int(c.value)}"
        tw = int(d.textlength(val, font=big))
        d.text((x + w // 2 - tw // 2, y + h // 2 - 12), val, font=big, fill="white" if c.value / vmax > 0.35 else "#0f172a")

    note = "Tip: For retrieval@K, define TP/FP/FN/TN at the segment level using your ground-truth timestamps."
    d.text((120, 630), note, font=small, fill="#475569")

    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="PNG")


def render_pr_at_k(pr_at_k: Dict[int, Dict[str, float]], out_path: str) -> None:
    """
    Render a compact 'matrix-like' table + mini-bars for Precision@K and Recall@K.
    """
    ks = sorted(pr_at_k.keys())
    W, H = 1100, 520
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    title_font = _font(20)
    small = _font(12)
    label = _font(14)

    title = "Retrieval quality by K (Precision@K / Recall@K)"
    d.text((W // 2 - int(d.textlength(title, font=title_font)) // 2, 20), title, font=title_font, fill="#0f172a")

    x0, y0 = 90, 90
    row_h = 62
    col_k = 70
    col_p = 340
    col_r = 340
    gap = 18

    # header
    d.text((x0, y0), "K", font=label, fill="#334155")
    d.text((x0 + col_k + gap, y0), "Precision", font=label, fill="#334155")
    d.text((x0 + col_k + gap + col_p + gap, y0), "Recall", font=label, fill="#334155")

    bar_h = 18
    for i, k in enumerate(ks):
        y = y0 + 40 + i * row_h
        p = float(pr_at_k[k].get("precision", 0.0))
        r = float(pr_at_k[k].get("recall", 0.0))

        d.text((x0, y), str(k), font=label, fill="#0f172a")

        # precision bar
        px = x0 + col_k + gap
        d.rounded_rectangle((px, y + 6, px + col_p, y + 6 + bar_h), radius=10, fill="#f1f5f9", outline="#cbd5e1", width=1)
        d.rounded_rectangle((px, y + 6, px + int(col_p * max(0.0, min(1.0, p))), y + 6 + bar_h), radius=10, fill="#1e3a8a")
        d.text((px + col_p + 10, y + 2), f"{p:.2f}", font=small, fill="#0f172a")

        # recall bar
        rx = x0 + col_k + gap + col_p + gap
        d.rounded_rectangle((rx, y + 6, rx + col_r, y + 6 + bar_h), radius=10, fill="#f1f5f9", outline="#cbd5e1", width=1)
        d.rounded_rectangle((rx, y + 6, rx + int(col_r * max(0.0, min(1.0, r))), y + 6 + bar_h), radius=10, fill="#0f766e")
        d.text((rx + col_r + 10, y + 2), f"{r:.2f}", font=small, fill="#0f172a")

    footer = "Replace the example numbers by exporting your eval results as JSON and re-running this script."
    d.text((90, H - 40), footer, font=small, fill="#475569")

    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="PNG")


def main() -> None:
    # Example numbers (placeholder) — swap these with real eval outputs later.
    # Confusion matrix here is illustrative; retrieval evaluation is usually computed per-query with TP/FP/FN.
    render_confusion_matrix(tp=87, fp=13, fn=18, tn=82, out_path="docs/evaluation/confusion_matrix.png")

    pr_at_k = {
        1: {"precision": 0.78, "recall": 0.45},
        3: {"precision": 0.85, "recall": 0.72},
        5: {"precision": 0.87, "recall": 0.82},
    }
    render_pr_at_k(pr_at_k, out_path="docs/evaluation/pr_at_k.png")

    # Also export the example inputs so you can edit them easily.
    _ensure_dir("docs/evaluation")
    with open("docs/evaluation/example_eval_inputs.json", "w", encoding="utf-8") as f:
        json.dump({"confusion_matrix": {"tp": 87, "fp": 13, "fn": 18, "tn": 82}, "pr_at_k": pr_at_k}, f, indent=2)

    print("Wrote docs/evaluation/confusion_matrix.png and pr_at_k.png")


if __name__ == "__main__":
    main()


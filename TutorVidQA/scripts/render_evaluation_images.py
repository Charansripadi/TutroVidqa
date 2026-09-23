from __future__ import annotations

import os
from typing import Dict

from PIL import Image, ImageDraw, ImageFont


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def render_metric_bars(metrics: Dict[str, float], out_path: str) -> None:
    names = list(metrics.keys())
    vals = [float(metrics[k]) for k in names]

    W, H = 1400, 650
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(20)
    label_font = _font(14)
    small = _font(12)

    title = "TutorVidQA: Example Evaluation Metrics"
    d.text((W // 2 - int(d.textlength(title, font=title_font)) // 2, 20), title, font=title_font, fill="#0f172a")

    plot_x0, plot_y0 = 80, 90
    plot_x1, plot_y1 = W - 40, H - 140

    # axes
    d.line((plot_x0, plot_y1, plot_x1, plot_y1), fill="#111827", width=2)
    d.line((plot_x0, plot_y0, plot_x0, plot_y1), fill="#111827", width=2)
    d.text((20, (plot_y0 + plot_y1) // 2), "Score (0–1)", font=label_font, fill="#334155")

    # grid
    for i in range(0, 11, 2):
        y = plot_y1 - int((plot_y1 - plot_y0) * (i / 10.0))
        d.line((plot_x0, y, plot_x1, y), fill="#e2e8f0", width=1)
        d.text((plot_x0 - 35, y - 6), f"{i/10:.1f}", font=small, fill="#64748b")

    n = len(names)
    gap = 18
    bar_w = int((plot_x1 - plot_x0 - gap * (n + 1)) / n)
    color = "#1e3a8a"

    for i, (name, v) in enumerate(zip(names, vals)):
        x0 = plot_x0 + gap + i * (bar_w + gap)
        x1 = x0 + bar_w
        y1 = plot_y1
        y0 = plot_y1 - int((plot_y1 - plot_y0) * max(0.0, min(1.0, v)))
        d.rectangle((x0, y0, x1, y1), fill=color, outline="#0f172a", width=1)
        d.text((x0 + 4, y0 - 18), f"{v:.2f}", font=small, fill="#0f172a")
        # x labels (stacked)
        d.text((x0, plot_y1 + 12), name, font=small, fill="#334155")

    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="PNG")


def render_latency_plot(latency_s: float, out_path: str) -> None:
    W, H = 900, 550
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(20)
    small = _font(12)

    title = "TutorVidQA: Retrieval Latency (example)"
    d.text((W // 2 - int(d.textlength(title, font=title_font)) // 2, 20), title, font=title_font, fill="#0f172a")

    plot_x0, plot_y0 = 90, 90
    plot_x1, plot_y1 = W - 40, H - 120

    d.line((plot_x0, plot_y1, plot_x1, plot_y1), fill="#111827", width=2)
    d.line((plot_x0, plot_y0, plot_x0, plot_y1), fill="#111827", width=2)
    d.text((20, (plot_y0 + plot_y1) // 2), "Seconds", font=_font(14), fill="#334155")

    # scale to 0..max(2s, latency+0.5)
    vmax = max(2.0, float(latency_s) + 0.5)
    for t in [0.0, vmax / 2, vmax]:
        y = plot_y1 - int((plot_y1 - plot_y0) * (t / vmax))
        d.line((plot_x0, y, plot_x1, y), fill="#e2e8f0", width=1)
        d.text((plot_x0 - 50, y - 6), f"{t:.1f}", font=small, fill="#64748b")

    bar_x0 = plot_x0 + 220
    bar_x1 = plot_x0 + 520
    bar_y1 = plot_y1
    bar_y0 = plot_y1 - int((plot_y1 - plot_y0) * (float(latency_s) / vmax))
    d.rectangle((bar_x0, bar_y0, bar_x1, bar_y1), fill="#0f766e", outline="#0f172a", width=1)
    d.text((bar_x0, plot_y1 + 12), "Retrieval latency", font=small, fill="#334155")
    d.text((bar_x0 + 90, bar_y0 - 18), f"{float(latency_s):.2f}s", font=small, fill="#0f172a")

    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="PNG")


def main() -> None:
    # These mirror the example table currently shown in README.md.
    metrics = {
        "Precision@5": 0.87,
        "Recall@5": 0.82,
        "F1": 0.84,
        "Citation Acc.": 0.86,
        "Temporal Grounding": 0.79,
        "Hallucination (↓)": 0.11,
        "BLEU": 0.76,
    }
    render_metric_bars(metrics, "docs/evaluation/metrics_bar.png")
    render_latency_plot(1.40, "docs/evaluation/latency.png")
    print("Wrote docs/evaluation/*.png")


if __name__ == "__main__":
    main()


## TutorVidQA – Figure descriptions (captions)

Use these as report captions or slide notes. File paths are relative to the repo root.

### Architecture figures

#### `docs/architecture/architecture_overview.png`

End-to-end TutorVidQA proof-of-concept architecture. The video is segmented into fixed windows; a representative keyframe is extracted per segment and audio is transcribed with Whisper. Transcript chunks are aligned to the same segment timestamps, then CLIP encodes segment text (and optionally frames) into embeddings stored in a FAISS vector index. A user question is encoded with the CLIP text encoder, used to retrieve the top‑K most relevant segments, and a grounded answer stub is produced with timestamp citations.

#### `docs/architecture/retrieval_flow.png`

Retrieval-Augmented QA flow at inference time. The question is embedded, a similarity search is run against FAISS, the top‑K timestamped segments are returned as evidence, and the answer is generated from that evidence with explicit citations. This figure emphasizes the “grounding” mechanism used to reduce hallucinations.

#### `docs/architecture/dataset_artifact_flow.png`

Dataset/artifact pipeline view showing the concrete files produced at each step and the scripts responsible for generating them. It documents the reproducible pipeline from raw videos to segment maps and keyframes, Whisper transcripts, CLIP embeddings, a global FAISS index, and optional QA-pair annotations for benchmarking.

### Evaluation figures

#### `docs/evaluation/metrics_bar.png`

Summary bar chart of key evaluation metrics (example values). Intended as a “headline results” figure for retrieval quality and grounding quality (e.g., Precision@K/Recall@K/F1, citation accuracy, temporal grounding accuracy) plus an error-style metric such as hallucination rate (lower is better). Replace the example numbers with your measured results for your dataset/benchmark split.

#### `docs/evaluation/latency.png`

Single-number retrieval latency visualization (example value). Useful for a quick performance summary, but best interpreted alongside the latency breakdown figure that separates segmentation/transcription/embedding/indexing/retrieval costs.

#### `docs/evaluation/confusion_matrix.png`

Confusion-matrix style visualization for a chosen evaluation setup (e.g., retrieval@K). It illustrates counts of correct retrievals (TP), incorrect retrievals (FP), missed relevant segments (FN), and correctly rejected irrelevant segments (TN). For TutorVidQA, “positive” typically means a retrieved segment overlaps the ground-truth timestamp window (or matches the ground-truth segment id) under a defined overlap threshold.

#### `docs/evaluation/pr_at_k.png`

Matrix-like summary of Precision@K and Recall@K at a small set of K values (e.g., K=1,3,5). This is a compact alternative to curves and is easy to interpret in a report table/figure hybrid. Replace the example values with results computed over your evaluation queries.

#### `docs/evaluation/pr_curves.png`

Precision@K and Recall@K curves across \(K\) (e.g., 1–10). This figure shows the trade-off between returning more evidence segments (increasing recall) and maintaining relevance in the retrieved set (precision). Curves are preferred when you want to justify a specific choice of K.

#### `docs/evaluation/latency_breakdown.png`

End-to-end latency decomposition by pipeline stage (example values). This figure helps explain where time is spent (segmentation, Whisper transcription, CLIP embedding, FAISS indexing, retrieval) and supports claims about system efficiency and optimization targets.

#### `docs/evaluation/ablation.png`

Ablation study template comparing retrieval configurations (example: text-only vs image-only vs text+image). Use it to demonstrate which modality or design choice contributes most to retrieval quality, and to justify the final architecture.

#### `docs/evaluation/qualitative_examples.png`

Qualitative evaluation template for case studies. Each card pairs a question with ground truth timestamps/segment, retrieved timestamps/segment, and a short evidence snippet. Include both successes and failure cases to demonstrate grounding behavior, typical errors, and limitations.


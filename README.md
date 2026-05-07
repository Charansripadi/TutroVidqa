## TutroVidqa (TutorVidQA / EduVidQA)

Research-oriented pipeline to build an educational Video Question Answering dataset (**EduVidQA**) from long-form tutorial/lecture videos.

- Project code and documentation live in `TutorVidQA/`.
- Includes download (yt-dlp), segmentation, Whisper transcription+alignment, CLIP embeddings, FAISS indexing, QA-pair generation, and dataset statistics.

### Quick start

```bash
cd TutorVidQA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_youtube.py --url "https://youtube.com/..."
```


import argparse

from video_segmenter import segment_video
from transcriber import transcribe_video
from embedder import create_embeddings
from retriever import retrieve_segments


def generate_stub_answer(results, question):
    """
    Creates a simple grounded answer stub.

    Args:
        results (list): Retrieved segments.
        question (str): User question.

    Returns:
        str: Grounded response.
    """
    if not results:
        return "No relevant content found."

    best = results[0]

    return (
        f"Based on timestamps {best['start']}s to {best['end']}s, "
        f"the lecture discusses: {best['text']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--question", required=True)

    args = parser.parse_args()

    print("[1] Segmenting video...")
    segments = segment_video(args.video)

    print("[2] Transcribing video...")
    transcripts = transcribe_video(args.video, segments)

    print("[3] Creating embeddings...")
    index, metadata = create_embeddings(transcripts, segments)

    print("[4] Retrieving relevant segments...")
    results = retrieve_segments(args.question, index, metadata)

    print("\nRetrieved Segments:\n")

    for item in results:
        print(f"Segment {item['segment_id']} | {item['start']}s - {item['end']}s")
        print(f"Text: {item['text']}\n")

    answer = generate_stub_answer(results, args.question)

    print("Grounded Answer:\n")
    print(answer)


"""
Process YouTube transcriptions into training data.
Extracts clean text from VTT subtitle files, then creates
conversational training samples based on David's voice/style.
"""

import os
import re
import json
import random
from pathlib import Path


def clean_vtt(vtt_path):
    """Extract clean text from a VTT subtitle file."""
    with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Remove VTT headers
    lines = content.split("\n")
    text_lines = []
    seen = set()

    for line in lines:
        line = line.strip()
        # Skip headers, timestamps, empty lines
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}", line):
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        # Remove HTML/VTT tags
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\[.*?\]", "", line)  # remove [Music] etc
        line = line.strip()
        if not line:
            continue
        # Deduplicate (VTT often repeats lines)
        if line not in seen:
            seen.add(line)
            text_lines.append(line)

    return " ".join(text_lines)


def extract_video_title(filename):
    """Extract human-readable title from VTT filename."""
    name = Path(filename).stem
    # Remove video ID prefix and .en suffix
    name = re.sub(r"^[A-Za-z0-9_-]+_", "", name, count=1)
    name = name.replace(".en", "")
    # Clean up unicode artifacts
    name = name.replace("｜", "|").replace("：", ":").replace("？", "?")
    return name


def create_training_samples(transcripts):
    """
    Create training data from transcripts.

    Formats:
    1. Raw transcript chunks (teaches the model David's vocabulary and style)
    2. Topic discussions (reformatted as natural conversation)
    3. Q&A pairs based on the content
    """
    samples = []

    for title, text in transcripts:
        if len(text.strip()) < 100:
            continue

        # --- Format 1: Raw transcript chunks ---
        # Split into paragraphs of ~200-500 words
        words = text.split()
        chunk_size = 300
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 100:
                samples.append({
                    "text": chunk,
                    "source": "youtube_raw",
                    "title": title,
                })

        # --- Format 2: Topic discussion ---
        # Frame the transcript as David explaining something
        if len(words) > 50:
            intro = " ".join(words[:150])
            samples.append({
                "text": f"Topic: {title}\n\n{intro}",
                "source": "youtube_topic",
                "title": title,
            })

        # --- Format 3: Conversational style ---
        # Extract sentences and create natural conversation flows
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Group into conversational exchanges
        for i in range(0, len(sentences) - 3, 3):
            exchange = ". ".join(sentences[i:i+3]) + "."
            if len(exchange) > 50:
                samples.append({
                    "text": exchange,
                    "source": "youtube_conversation",
                    "title": title,
                })

    return samples


def main():
    youtube_dir = Path("data/youtube")
    output_dir = Path("data/personal")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all VTT files
    vtt_files = sorted(youtube_dir.glob("*.vtt"))
    print(f"Found {len(vtt_files)} VTT files")

    transcripts = []
    all_text = []

    for vtt_path in vtt_files:
        title = extract_video_title(vtt_path.name)
        text = clean_vtt(vtt_path)
        if text:
            transcripts.append((title, text))
            all_text.append(text)
            print(f"  {title}: {len(text.split())} words")

    # Save combined raw transcripts
    raw_path = output_dir / "all_transcripts.txt"
    with open(raw_path, "w") as f:
        for title, text in transcripts:
            f.write(f"=== {title} ===\n\n{text}\n\n")

    total_words = sum(len(t.split()) for _, t in transcripts)
    print(f"\nTotal: {len(transcripts)} videos, {total_words:,} words")

    # Create training samples
    samples = create_training_samples(transcripts)
    print(f"Created {len(samples)} training samples")

    # Save as JSONL for training
    samples_path = output_dir / "training_samples.jsonl"
    with open(samples_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Save stats
    stats = {
        "num_videos": len(transcripts),
        "total_words": total_words,
        "num_samples": len(samples),
        "sample_types": {
            "youtube_raw": sum(1 for s in samples if s["source"] == "youtube_raw"),
            "youtube_topic": sum(1 for s in samples if s["source"] == "youtube_topic"),
            "youtube_conversation": sum(1 for s in samples if s["source"] == "youtube_conversation"),
        },
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  all_transcripts.txt: {raw_path.stat().st_size / 1024:.1f} KB")
    print(f"  training_samples.jsonl: {len(samples)} samples")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

"""
Prepare SFT Data — Downloads and formats open SFT datasets
===========================================================
Downloads SmolTalk from HuggingFace and combines it with our custom
SFT data into a single training file.

Sources:
  - SmolTalk (HuggingFaceTB/smoltalk, ~1M convos) → filter to ~100k high quality
  - Our custom diverse data (data/personal/sft_diverse_only.jsonl, ~2k convos)
  - Our original tech SFT (data/personal/sft_conversations.jsonl, ~6k convos) → sample 2k

Output:
  - data/sft_smoltalk.jsonl      (~100k SmolTalk conversations)
  - data/sft_combined.jsonl      (~104k combined for SFT training)

Usage:
    python prepare_sft_data.py
    python prepare_sft_data.py --max-smoltalk 50000    # smaller subset
    python prepare_sft_data.py --skip-download          # use existing smoltalk file
"""

import os
import sys
import json
import random
import hashlib
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SMOLTALK_DATASET = "HuggingFaceTB/smoltalk"
SMOLTALK_OUTPUT = "data/sft_smoltalk.jsonl"
COMBINED_OUTPUT = "data/sft_combined.jsonl"

CUSTOM_DIVERSE_PATH = "data/personal/sft_diverse_only.jsonl"
CUSTOM_TECH_PATH = "data/personal/sft_conversations.jsonl"

MAX_SMOLTALK = 100_000
TECH_SAMPLE = 2_000
MIN_TOKENS_APPROX = 50   # skip very short conversations (rough char estimate)
MAX_TOKENS_APPROX = 900  # skip very long ones (rough char estimate, ~4 chars/token)


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# SmolTalk format conversion
# ---------------------------------------------------------------------------

def convert_smoltalk_to_hamner(messages):
    """Convert SmolTalk message list to our <|role|> format.

    SmolTalk format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    Our format: "<|system|>\\n...\\n<|user|>\\n...\\n<|assistant|>\\n..."
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            return None  # skip broken conversations
        if role in ("system", "user", "assistant"):
            parts.append(f"<|{role}|>\n{content}")
        # Skip unknown roles

    if not parts:
        return None

    # Must have at least one user + one assistant turn
    roles = [msg["role"] for msg in messages if msg.get("role")]
    if "user" not in roles or "assistant" not in roles:
        return None

    return "\n".join(parts)


def estimate_tokens(text):
    """Rough token count estimate (~4 chars per token for English)."""
    return len(text) / 4


def dedup_key(text):
    """Extract first user message for deduplication."""
    if "<|user|>\n" in text:
        after = text.split("<|user|>\n", 1)[1]
        if "<|assistant|>" in after:
            first_msg = after.split("<|assistant|>", 1)[0].strip()
        else:
            first_msg = after.strip()
        return hashlib.md5(first_msg.lower().encode()).hexdigest()
    return hashlib.md5(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Download and process SmolTalk
# ---------------------------------------------------------------------------

def download_smoltalk(max_samples=MAX_SMOLTALK, output_path=SMOLTALK_OUTPUT):
    from datasets import load_dataset

    log(f"Downloading SmolTalk from {SMOLTALK_DATASET}...")
    log(f"Target: {max_samples:,} high-quality conversations")

    # SmolTalk has multiple subsets — load all and sample
    dataset = load_dataset(SMOLTALK_DATASET, "all", split="train", streaming=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seen_keys = set()
    kept = 0
    skipped_short = 0
    skipped_long = 0
    skipped_broken = 0
    skipped_dup = 0
    total_processed = 0

    with open(output_path, "w") as f:
        for sample in dataset:
            total_processed += 1

            if total_processed % 50000 == 0:
                log(f"  Processed {total_processed:,} | Kept {kept:,} | "
                    f"Skipped: {skipped_short} short, {skipped_long} long, "
                    f"{skipped_broken} broken, {skipped_dup} dup")

            if kept >= max_samples:
                break

            # Get messages
            messages = sample.get("messages", [])
            if not messages:
                skipped_broken += 1
                continue

            # Convert format
            text = convert_smoltalk_to_hamner(messages)
            if text is None:
                skipped_broken += 1
                continue

            # Length filter
            approx_tokens = estimate_tokens(text)
            if approx_tokens < MIN_TOKENS_APPROX:
                skipped_short += 1
                continue
            if approx_tokens > MAX_TOKENS_APPROX:
                skipped_long += 1
                continue

            # Dedup
            key = dedup_key(text)
            if key in seen_keys:
                skipped_dup += 1
                continue
            seen_keys.add(key)

            # Write
            f.write(json.dumps({"text": text}) + "\n")
            kept += 1

    log(f"SmolTalk processing complete:")
    log(f"  Total processed: {total_processed:,}")
    log(f"  Kept: {kept:,}")
    log(f"  Skipped: {skipped_short} short, {skipped_long} long, "
        f"{skipped_broken} broken, {skipped_dup} dup")
    log(f"  Output: {output_path}")
    return kept


# ---------------------------------------------------------------------------
# Load custom data
# ---------------------------------------------------------------------------

def load_custom_data(path, max_samples=None, label="custom"):
    """Load our custom JSONL data files."""
    if not os.path.exists(path):
        log(f"  {label}: not found at {path}")
        return []

    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                samples.append(entry["text"])

    log(f"  {label}: loaded {len(samples)} conversations from {path}")

    if max_samples and len(samples) > max_samples:
        random.seed(42)
        samples = random.sample(samples, max_samples)
        log(f"  {label}: sampled down to {len(samples)}")

    return samples


# ---------------------------------------------------------------------------
# Combine all data
# ---------------------------------------------------------------------------

def combine_data(smoltalk_path=SMOLTALK_OUTPUT, output_path=COMBINED_OUTPUT):
    log("\nCombining all SFT data...")

    all_samples = []

    # SmolTalk
    if os.path.exists(smoltalk_path):
        with open(smoltalk_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    all_samples.append(("smoltalk", entry["text"]))
        log(f"  SmolTalk: {sum(1 for s in all_samples if s[0] == 'smoltalk'):,}")
    else:
        log(f"  WARNING: SmolTalk file not found at {smoltalk_path}")

    # Custom diverse
    for text in load_custom_data(CUSTOM_DIVERSE_PATH, label="custom_diverse"):
        all_samples.append(("custom_diverse", text))

    # Custom tech (sampled)
    for text in load_custom_data(CUSTOM_TECH_PATH, max_samples=TECH_SAMPLE,
                                  label="custom_tech"):
        all_samples.append(("custom_tech", text))

    # Shuffle
    random.seed(42)
    random.shuffle(all_samples)

    # Write combined
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for source, text in all_samples:
            f.write(json.dumps({"text": text, "source": source}) + "\n")

    # Stats
    source_counts = {}
    for source, _ in all_samples:
        source_counts[source] = source_counts.get(source, 0) + 1

    log(f"\nCombined dataset: {len(all_samples):,} conversations")
    for source, count in sorted(source_counts.items()):
        log(f"  {source}: {count:,} ({count/len(all_samples)*100:.1f}%)")
    log(f"Output: {output_path}")

    return len(all_samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare SFT Data")
    parser.add_argument("--max-smoltalk", type=int, default=MAX_SMOLTALK,
                        help=f"Max SmolTalk conversations (default: {MAX_SMOLTALK})")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip SmolTalk download, use existing file")
    args = parser.parse_args()

    log("=" * 70)
    log("PREPARE SFT DATA")
    log("=" * 70)

    if not args.skip_download:
        download_smoltalk(max_samples=args.max_smoltalk)
    else:
        if os.path.exists(SMOLTALK_OUTPUT):
            count = sum(1 for _ in open(SMOLTALK_OUTPUT))
            log(f"Using existing SmolTalk file: {count:,} conversations")
        else:
            log("WARNING: --skip-download but no SmolTalk file found!")

    total = combine_data()

    log("\n" + "=" * 70)
    log(f"DONE — {total:,} total conversations ready for SFT")
    log("=" * 70)


if __name__ == "__main__":
    main()

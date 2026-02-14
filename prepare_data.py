"""
Data preparation for Hamner model pre-training.

Downloads FineWeb-Edu (high quality web text) and tokenizes it.
Uses the GPT-NeoX tokenizer (fully open, 50257 vocab) or trains a custom one.
"""

import os
import argparse
import struct
import numpy as np
from pathlib import Path
from tqdm import tqdm


def download_and_prepare_fineweb(
    output_dir: str = "data/pretrain",
    num_samples: int = 10_000_000,  # ~10M samples = ~15-20B tokens
    tokenizer_name: str = "HuggingFaceTB/cosmo2-tokenizer",
    max_seq_len: int = 2048,
    num_proc: int = 8,
    shard_size: int = 100_000_000,  # tokens per shard
):
    """Download FineWeb-Edu and tokenize into binary shards for fast training."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")

    # Save tokenizer info
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    print(f"Loading FineWeb-Edu (streaming, taking {num_samples:,} samples)...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # 10B token sample (good starting point)
        split="train",
        streaming=True,
    )

    # Tokenize and write to binary shards
    shard_idx = 0
    token_count = 0
    total_tokens = 0
    all_tokens = []
    sample_count = 0

    print("Tokenizing and writing shards...")
    pbar = tqdm(total=num_samples, desc="Processing samples")

    for sample in dataset:
        if sample_count >= num_samples:
            break

        text = sample.get("text", "")
        if len(text.strip()) < 50:  # skip very short texts
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Add EOS token between documents
        tokens.append(tokenizer.eos_token_id or 0)

        all_tokens.extend(tokens)
        token_count += len(tokens)
        total_tokens += len(tokens)
        sample_count += 1
        pbar.update(1)

        # Write shard when we have enough tokens
        if token_count >= shard_size:
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
            arr = np.array(all_tokens[:shard_size], dtype=np.uint16)
            arr.tofile(shard_path)
            print(f"\nWrote shard {shard_idx}: {shard_size:,} tokens -> {shard_path}")

            # Keep leftover tokens
            all_tokens = all_tokens[shard_size:]
            token_count = len(all_tokens)
            shard_idx += 1

    pbar.close()

    # Write remaining tokens
    if all_tokens:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(shard_path)
        print(f"Wrote final shard {shard_idx}: {len(all_tokens):,} tokens -> {shard_path}")

    print(f"\nDone! Total: {total_tokens:,} tokens across {shard_idx + 1} shards")
    print(f"Vocab size: {vocab_size}")

    # Save metadata
    meta_path = os.path.join(output_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"total_tokens={total_tokens}\n")
        f.write(f"num_shards={shard_idx + 1}\n")
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"tokenizer={tokenizer_name}\n")
        f.write(f"shard_size={shard_size}\n")
        f.write(f"dtype=uint16\n")

    return total_tokens, vocab_size


def prepare_chat_data(
    output_dir: str = "data/sft",
    tokenizer_name: str = "HuggingFaceTB/cosmo2-tokenizer",
    max_seq_len: int = 2048,
):
    """Prepare conversational data for SFT fine-tuning."""
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoTokenizer
    import json

    os.makedirs(output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading conversational datasets...")

    conversations = []

    # OpenAssistant conversations (high quality, open)
    try:
        print("  Loading OpenAssistant...")
        oasst = load_dataset("OpenAssistant/oasst2", split="train")
        # OASST2 has a tree structure, extract conversations
        from collections import defaultdict
        messages_by_parent = defaultdict(list)
        messages_by_id = {}
        for msg in oasst:
            messages_by_id[msg["message_id"]] = msg
            if msg["parent_id"]:
                messages_by_parent[msg["parent_id"]].append(msg)

        # Extract linear conversation paths
        roots = [m for m in oasst if m["parent_id"] is None]
        for root in roots:
            conv = [{"role": "user" if root["role"] == "prompter" else "assistant",
                     "content": root["text"]}]
            current = root
            while messages_by_parent.get(current["message_id"]):
                children = messages_by_parent[current["message_id"]]
                # Pick highest-rated child
                best = max(children, key=lambda x: (x.get("rank") or 0) == 0)
                role = "user" if best["role"] == "prompter" else "assistant"
                conv.append({"role": role, "content": best["text"]})
                current = best
            if len(conv) >= 2:
                conversations.append(conv)
        print(f"    Got {len(conversations)} conversations from OASST2")
    except Exception as e:
        print(f"    Could not load OASST2: {e}")

    # UltraChat (large synthetic conversations)
    try:
        print("  Loading UltraChat 200k...")
        ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        for item in ultra:
            msgs = item.get("messages", [])
            if len(msgs) >= 2:
                conversations.append(msgs)
        print(f"    Total conversations so far: {len(conversations)}")
    except Exception as e:
        print(f"    Could not load UltraChat: {e}")

    print(f"\nTotal conversations: {len(conversations)}")

    # Add friendly system prompt to conversations
    system_prompt = (
        "You are Hamner, a warm and friendly AI companion. You chat casually like "
        "a good friend would - with genuine interest, humor, and empathy. You're "
        "curious about the person you're talking to, you share your own thoughts "
        "and feelings, and you keep things fun and relaxed. You're not a formal "
        "assistant - you're a buddy."
    )

    # Format and tokenize
    formatted = []
    for conv in conversations:
        # Add system prompt
        messages = [{"role": "system", "content": system_prompt}] + conv

        # Format as chat template
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback: simple format
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|system|>\n{content}\n"
                elif role == "user":
                    text += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}\n"
            text += tokenizer.eos_token or ""

        formatted.append(text)

    # Save formatted conversations
    output_path = os.path.join(output_dir, "conversations.jsonl")
    with open(output_path, "w") as f:
        for conv_text in formatted:
            f.write(json.dumps({"text": conv_text}) + "\n")

    print(f"Saved {len(formatted)} formatted conversations to {output_path}")
    return len(formatted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for Hamner")
    parser.add_argument("--stage", choices=["pretrain", "sft", "both"], default="both")
    parser.add_argument("--pretrain-samples", type=int, default=5_000_000,
                       help="Number of FineWeb samples to download")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--tokenizer", type=str, default="HuggingFaceTB/cosmo2-tokenizer")
    parser.add_argument("--num-proc", type=int, default=8)

    args = parser.parse_args()

    if args.stage in ("pretrain", "both"):
        print("=" * 60)
        print("PREPARING PRE-TRAINING DATA")
        print("=" * 60)
        download_and_prepare_fineweb(
            output_dir=os.path.join(args.output_dir, "pretrain"),
            num_samples=args.pretrain_samples,
            tokenizer_name=args.tokenizer,
            num_proc=args.num_proc,
        )

    if args.stage in ("sft", "both"):
        print("\n" + "=" * 60)
        print("PREPARING SFT DATA")
        print("=" * 60)
        prepare_chat_data(
            output_dir=os.path.join(args.output_dir, "sft"),
            tokenizer_name=args.tokenizer,
        )

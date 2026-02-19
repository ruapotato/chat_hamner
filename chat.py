#!/usr/bin/env python3
"""
Chat with Al Hamner — a 164M param AI trained from scratch on one RTX 3090.

Usage:
    python chat.py                                  # uses SFT checkpoint
    python chat.py --checkpoint path/to/model.pt    # use specific checkpoint
    python chat.py --system "Custom system prompt"  # override personality
    python chat.py --cpu                            # run on CPU
"""

import os
import sys
import torch
import readline  # enables arrow keys + history in input()
from model import HamnerModel, HamnerConfig

# Defaults
DEFAULT_CHECKPOINT = "checkpoints/sft/latest.pt"
FALLBACK_CHECKPOINT = "checkpoints/pretrain_v2/latest.pt"
DEFAULT_SYSTEM_PROMPT = (
    "You are Al Hamner, a sharp-witted AI made by David Hamner. "
    "You're casual, funny, opinionated, and self-aware. "
    "You talk like a real person, not a corporate chatbot."
)
VOICE_SAMPLE_FILE = "data/personal/voice_sample.txt"
MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 256


def load_model(checkpoint_path, device="cuda"):
    """Load model from checkpoint."""
    print(f"  Loading {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = HamnerConfig(**ckpt["config"])
    model = HamnerModel(config).to(device)

    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    total_p, _ = model.count_parameters()
    step = ckpt.get("step", "?")
    loss = ckpt.get("avg_loss", None)
    training_type = ckpt.get("training_type", "pretrain")
    loss_str = f"{loss:.3f}" if isinstance(loss, float) else "?"
    print(f"  {total_p:,} params | {training_type} step {step} | loss {loss_str}")

    model.eval()
    return model, config


def load_system_prompt():
    """Load system prompt from voice_sample.txt if available."""
    if os.path.exists(VOICE_SAMPLE_FILE):
        with open(VOICE_SAMPLE_FILE) as f:
            text = f.read()
        if "<|system|>\n" in text:
            after = text.split("<|system|>\n", 1)[1]
            if "<|user|>" in after:
                return after.split("<|user|>", 1)[0].strip()
    return DEFAULT_SYSTEM_PROMPT


@torch.no_grad()
def generate_response(model, tokenizer, conversation_tokens, device="cuda",
                      max_new_tokens=MAX_NEW_TOKENS, temperature=0.8,
                      top_k=40, top_p=0.9):
    """Generate a response from conversation context."""
    input_ids = torch.tensor([conversation_tokens], dtype=torch.long, device=device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id or 0,
    )

    new_tokens = output[0][len(conversation_tokens):].tolist()
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Stop at turn boundaries or EOS
    for stop in ["<|user|>", "<|system|>", "<|endoftext|>"]:
        idx = response.find(stop)
        if idx != -1:
            response = response[:idx]

    return response.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chat with Al Hamner")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--system", type=str, default=None,
                        help="Override system prompt")
    parser.add_argument("--temp", type=float, default=0.8,
                        help="Generation temperature (default: 0.8)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        if os.path.exists(DEFAULT_CHECKPOINT):
            ckpt_path = DEFAULT_CHECKPOINT
        elif os.path.exists(FALLBACK_CHECKPOINT):
            ckpt_path = FALLBACK_CHECKPOINT
            print("  (no SFT checkpoint found, using pretrained base)")
        else:
            print("Error: no checkpoint found. Run train_sft.py first.")
            sys.exit(1)

    # Banner
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║         Chat with Al Hamner          ║")
    print("  ║  164M params · trained from scratch   ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    # Load model
    model, config = load_model(ckpt_path, device)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # System prompt
    system_prompt = args.system or load_system_prompt()
    system_tokens = tokenizer.encode(
        f"<|system|>\n{system_prompt}\n", add_special_tokens=False
    )

    # State
    conversation_tokens = list(system_tokens)
    temperature = args.temp
    max_context = MAX_SEQ_LEN - MAX_NEW_TOKENS

    print()
    print(f"  System: {system_prompt[:70]}...")
    print(f"  Device: {device} | Temp: {temperature}")
    print(f"  Commands: /quit /clear /temp N /system /history")
    print()

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] in ("/quit", "/exit", "/q"):
                print("Bye!")
                break
            elif cmd[0] == "/clear":
                conversation_tokens = list(system_tokens)
                print("  (conversation cleared)")
                continue
            elif cmd[0] == "/temp" and len(cmd) > 1:
                try:
                    temperature = float(cmd[1])
                    print(f"  (temperature = {temperature})")
                except ValueError:
                    print("  (invalid number)")
                continue
            elif cmd[0] == "/system":
                print(f"  {system_prompt}")
                continue
            elif cmd[0] == "/history":
                text = tokenizer.decode(conversation_tokens, skip_special_tokens=False)
                print(f"  ({len(conversation_tokens)} tokens)")
                # Show last ~500 chars
                if len(text) > 500:
                    print(f"  ...{text[-500:]}")
                else:
                    print(f"  {text}")
                continue
            else:
                print(f"  (unknown: {cmd[0]})")
                continue

        # Append user turn + assistant tag
        user_turn = tokenizer.encode(
            f"<|user|>\n{user_input}\n<|assistant|>\n", add_special_tokens=False
        )
        conversation_tokens.extend(user_turn)

        # Truncate old context if needed (keep system prompt + recent turns)
        if len(conversation_tokens) > max_context:
            # Keep system tokens at the front, trim middle
            overflow = len(conversation_tokens) - max_context
            conversation_tokens = system_tokens + conversation_tokens[len(system_tokens) + overflow:]

        # Generate
        response = generate_response(
            model, tokenizer, conversation_tokens, device,
            temperature=temperature,
        )

        if not response:
            response = "..."

        # Append assistant response to context
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        conversation_tokens.extend(response_tokens)

        # Print
        print(f"\033[1;33mAl:\033[0m  {response}")
        print()


if __name__ == "__main__":
    main()

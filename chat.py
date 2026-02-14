"""
Hamner Chat Interface
=====================
Interactive chat with a trained Hamner model.
"""

import os
import sys
import json
import torch
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from model import HamnerModel, HamnerConfig

console = Console()


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a trained model from checkpoint."""
    console.print(f"[bold blue]Loading model from {checkpoint_path}...[/bold blue]")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct config
    config_dict = ckpt["config"]
    config = HamnerConfig(**config_dict)

    # Create model
    model = HamnerModel(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    total_p, _ = model.count_parameters()
    variant = ckpt.get("variant_name", "unknown")
    step = ckpt.get("step", 0)
    avg_loss = ckpt.get("avg_loss", "N/A")

    console.print(f"[green]Loaded {variant} | {total_p:,} params | step {step} | loss {avg_loss}[/green]")

    return model, config


def chat_loop(model, config, tokenizer, device="cuda"):
    """Interactive chat loop."""

    console.print(Panel(
        "[bold magenta]Hamner Chat[/bold magenta]\n"
        "A friendly AI companion trained from scratch.\n"
        "Type 'quit' to exit, 'clear' to reset conversation.",
        title="Welcome!",
        border_style="magenta",
    ))

    system_prompt = (
        "You are Hamner, a warm and friendly AI companion. You chat casually like "
        "a good friend would - with genuine interest, humor, and empathy. You're "
        "curious about the person you're talking to."
    )

    conversation_tokens = tokenizer.encode(
        f"<|system|>\n{system_prompt}\n",
        add_special_tokens=False,
    )

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if user_input.strip().lower() == "quit":
            console.print("[yellow]Goodbye![/yellow]")
            break
        if user_input.strip().lower() == "clear":
            conversation_tokens = tokenizer.encode(
                f"<|system|>\n{system_prompt}\n",
                add_special_tokens=False,
            )
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if not user_input.strip():
            continue

        # Add user message
        user_tokens = tokenizer.encode(
            f"<|user|>\n{user_input}\n<|assistant|>\n",
            add_special_tokens=False,
        )
        conversation_tokens.extend(user_tokens)

        # Truncate if too long
        max_context = config.max_seq_len - 256  # leave room for generation
        if len(conversation_tokens) > max_context:
            conversation_tokens = conversation_tokens[-max_context:]

        # Generate
        input_ids = torch.tensor([conversation_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id or 2,
            )

        # Decode new tokens
        new_tokens = output_ids[0, len(conversation_tokens):].tolist()
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Clean up response (stop at user/system markers)
        for stop in ["<|user|>", "<|system|>", "<|assistant|>"]:
            if stop in response:
                response = response[:response.index(stop)]
        response = response.strip()

        # Update conversation
        assistant_tokens = tokenizer.encode(response, add_special_tokens=False)
        assistant_tokens.append(tokenizer.eos_token_id or 2)
        conversation_tokens.extend(assistant_tokens)

        console.print(f"\n[bold magenta]Hamner:[/bold magenta] {response}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chat with Hamner")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--tokenizer", type=str, default="HuggingFaceTB/cosmo2-tokenizer")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, config = load_model(args.checkpoint, device=args.device)
    chat_loop(model, config, tokenizer, device=args.device)


if __name__ == "__main__":
    main()

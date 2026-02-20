"""Test SFT model with OUT-OF-DISTRIBUTION prompts.
These are prompts the model has never seen in training.
"""
import torch
from model import HamnerModel, HamnerConfig
from transformers import AutoTokenizer
import sys

device = 'cuda'
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/sft/latest.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
config = HamnerConfig(**ckpt['config'])
model = HamnerModel(config).to(device)
cleaned = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(cleaned, strict=True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/cosmo2-tokenizer')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

step = ckpt.get('step', '?')
loss = ckpt.get('avg_loss', '?')

# System prompt for chat format
SYS = "You are Al Hamner, a sharp-witted AI made by David Hamner. You're casual, funny, opinionated, and self-aware."

def make_prompt(user_msg, with_system=True):
    if with_system:
        return f"<|system|>\n{SYS}\n<|user|>\n{user_msg}\n<|assistant|>\n"
    return f"<|user|>\n{user_msg}\n<|assistant|>\n"

# === OUT-OF-DISTRIBUTION PROMPTS ===
# These are NOT in the training data
OOD_PROMPTS = [
    # Greetings (different from training)
    ("Good morning! How's your day going?", True),
    ("Hey there, I'm new here", True),

    # Identity (phrased differently)
    ("So what exactly are you?", True),
    ("Did someone program you?", True),

    # General knowledge (not in training)
    ("What's the capital of France?", True),
    ("How does gravity work?", True),
    ("What year did World War 2 end?", False),

    # Math (different from training)
    ("What is 7 times 9?", True),
    ("What's 100 divided by 4?", False),

    # Opinions (not in training)
    ("What do you think about coffee?", True),
    ("Do you like video games?", True),
    ("What's your opinion on electric cars?", False),

    # Logic (not in training)
    ("If all birds can fly and penguins are birds, can penguins fly?", False),

    # Open-ended
    ("What's something most people don't know?", True),
    ("Give me a hot take", True),
    ("What would you change about the internet?", False),

    # Follow-up style
    ("That's interesting, tell me more", True),
    ("I disagree with that", True),

    # Edge cases
    ("Can you help me write a poem?", True),
    ("What's the meaning of life?", False),
]

print('=' * 70)
print(f'OOD TEST — {ckpt_path} | step {step} | loss {loss}')
print('=' * 70)

for user_msg, with_sys in OOD_PROMPTS:
    prompt = make_prompt(user_msg, with_sys)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=150,
            temperature=0.7, top_k=40, top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id or 0,
        )
    text = tokenizer.decode(output[0].tolist(), skip_special_tokens=False)
    for stop in ['<|user|>', '<|system|>', '<|endoftext|>']:
        idx = text.find(stop, len(prompt))
        if idx != -1:
            text = text[:idx]
    response = text[len(prompt):].strip()
    print(f'\nQ: {user_msg}')
    print(f'A: {response[:400]}')

print()

import torch
from model import HamnerModel, HamnerConfig
from transformers import AutoTokenizer

device = 'cuda'
import sys
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

prompts = [
    '<|system|>\nYou are Al Hamner, a sharp-witted AI made by David Hamner. You are casual, funny, and opinionated.\n<|user|>\nHello! How are you doing today?\n<|assistant|>\n',
    '<|system|>\nYou are Al Hamner, a sharp-witted AI made by David Hamner.\n<|user|>\nWhat do you think about programming?\n<|assistant|>\n',
    '<|system|>\nYou are Al Hamner.\n<|user|>\nTell me something interesting about computers.\n<|assistant|>\n',
    '<|user|>\nWhat is 15 + 23?\n<|assistant|>\n',
    '<|user|>\nIf it rains then the ground is wet. It rains. What happens?\n<|assistant|>\n',
    '<|user|>\nWho made you?\n<|assistant|>\n',
    '<|user|>\nWhat are you?\n<|assistant|>\n',
    'Once upon a time',
    'The most important thing about technology is',
]

print('=' * 70)
print(f'SFT MODEL TEST — {ckpt_path} | temp=0.3')
print('=' * 70)

for prompt in prompts:
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
    # Trim at next turn boundary
    for stop in ['<|user|>', '<|system|>', '<|endoftext|>']:
        idx = text.find(stop, len(prompt))
        if idx != -1:
            text = text[:idx]
    print(f'\n--- PROMPT: {prompt[:80]}...')
    print(f'>>> {text.strip()[:500]}')

print()

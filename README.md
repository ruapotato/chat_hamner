# Chat Hamner

A conversational AI trained from scratch on a single RTX 3090. 164M parameters, custom transformer architecture, no pretrained weights.

Not a coding assistant. Not a knowledge oracle. Just **Al Hamner** — sharp-witted, opinionated, and built from zero.

## Sample Conversation

```
You: hello!
Al:  What's up! Ready when you are.

You: who made you?
Al:  David Hamner built me from scratch. Custom transformer architecture,
     trained on web text, conversations, and personal writings.

You: what is 15 + 23?
Al:  38.

You: what's the best programming language?
Al:  Python for versatility, Rust for safety, JavaScript for ubiquity.
     If I had to pick one: Python.
```

## How to Chat

```bash
python chat.py                                  # uses best SFT checkpoint
python chat.py --checkpoint path/to/model.pt    # use specific checkpoint
python chat.py --temp 1.0                       # adjust creativity
python chat.py --cpu                            # run on CPU
```

Commands inside chat: `/quit` `/clear` `/temp N` `/system` `/history`

## Architecture

164M parameter transformer built from scratch with modern components:

| Parameter | Value |
|-----------|-------|
| Hidden size | 768 |
| Layers | 20 |
| Attention heads | 12 (4 KV, GQA 3:1) |
| MLP | Dense SwiGLU |
| Intermediate size | 2048 |
| Parameters | 163.6M |
| Sequence length | 1024 |
| Tokenizer | cosmo2-tokenizer (49,152 vocab) |

Key components:
- **RMSNorm** pre-normalization
- **SwiGLU** gated activation
- **RoPE** rotary positional embeddings
- **GQA** grouped query attention
- **fp16** mixed precision training
- **torch.compile** for speed
- **Gradient checkpointing** for VRAM efficiency

The architecture was selected through a [tournament of 10 competing designs](#tournament-architecture-search).

## Training Pipeline

Al Hamner is trained in 3 stages. Total training time: ~20 hours on a single RTX 3090.

### Stage 1: Base Pretraining (164M, ~16 hours)

Train the base language model on FineWeb-Edu (educational web text).

```bash
python train.py --fresh
```

- **Data**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 10B token dataset (streamed)
- **Steps**: 106,000 (~2.6B tokens)
- **Final loss**: 3.08
- **LR**: 6e-4 with cosine decay
- **Batch size**: 24 x 1024 tokens
- **Output**: `checkpoints/pretrain_v2/latest.pt`

After this stage the model produces coherent English paragraphs but has no dialogue ability.

### Stage 2: Chat Pretraining (~3 hours)

Continue the base model on a dialogue-heavy data mix so it learns conversational format as natural language (no loss masking — raw next-token prediction on everything).

```bash
python train_chat_pretrain.py
```

- **Base**: `checkpoints/pretrain_v2/latest.pt`
- **Data mix**: 35% FineWeb + 35% SFT conversations + 15% personal voice + 10% TinyStories + 5% synthetic tasks
- **Steps**: ~20,000 (~0.5B tokens)
- **Final loss**: ~1.42
- **LR**: 5e-5 with cosine decay
- **Batch size**: 24 x 1024 tokens
- **Output**: `checkpoints/chat_pretrain/latest.pt`

After this stage the model can generate conversational text but can't condition responses on specific questions.

### Stage 3: Supervised Fine-Tuning (~10 minutes)

Fine-tune on 2,129 diverse conversations with loss computed only on assistant response tokens.

```bash
# Generate the SFT data
python generate_sft_v3.py

# Train (uses chat_pretrain checkpoint as base)
python train_sft.py --checkpoint checkpoints/chat_pretrain/latest.pt \
                    --data data/personal/sft_diverse_only.jsonl \
                    --epochs 5 --lr 3e-5
```

- **Base**: `checkpoints/chat_pretrain/latest.pt`
- **Data**: 2,129 conversations covering greetings, identity, math (all times tables 2-12), factual Q&A, opinions, reasoning, chitchat, follow-ups, edge cases
- **Epochs**: 5 (best checkpoint typically at epoch 2-3)
- **LR**: 3e-5 with linear warmup
- **Batch size**: 8
- **Output**: `checkpoints/sft/best.pt`

### SFT Data Breakdown

The diverse SFT data (`generate_sft_v3.py`) covers:

| Category | Base Count | Description |
|----------|-----------|-------------|
| Greetings | 59 | Hello, hey, hi, good morning, etc. |
| Identity | 23 | Who are you, who made you, what can you do |
| Math | 233 | ALL multiplication tables 2-12, addition, subtraction, division |
| Factual Q&A | 50 | Capital of France, speed of light, who wrote X |
| Opinions | 30 | Best language, AI ethics, tech takes |
| Chitchat | 28 | Weather, hobbies, jokes, small talk |
| Reasoning | 22 | Logic puzzles, cause/effect |
| Tech | 25 | Programming concepts, tools |
| Edge cases | 12 | Empty input, gibberish, adversarial |
| Follow-ups | 30 | "Tell me more", "I disagree", "thanks" |
| Multi-turn | 8 | Full conversations with context |

Data is augmented with rephrasing (2x all categories, extra 3x for greetings/identity/follow-ups, extra 1x for math) to reach 2,129 total conversations.

## Training Data

- **[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)** — 10B token educational web text (base pretraining)
- **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** — Simple narrative English (chat pretraining mix)
- **SFT conversations** — 6,010 tech-focused chat conversations (chat pretraining mix)
- **Diverse SFT** — 2,129 conversations covering all interaction types (SFT stage)
- **Synthetic tasks** — Arithmetic, counting, sorting, brackets, listops, copy/repeat
- **Personal voice** — YouTube transcriptions from [@davidhamner](https://www.youtube.com/@davidhamner)

## Reproduce from Scratch

```bash
# 1. Setup
git clone https://github.com/davidhamner/chat_hamner.git
cd chat_hamner
pip install -r requirements.txt

# 2. Process YouTube data (optional — for personal voice)
yt-dlp --write-auto-sub --sub-lang en --skip-download \
    -o "data/youtube/%(id)s_%(title)s" "https://www.youtube.com/@davidhamner"
python process_youtube.py

# 3. Generate SFT data
python generate_sft_data.py      # generates 6,010 tech conversations
python generate_sft_v3.py        # generates 2,129 diverse conversations

# 4. Stage 1: Base pretrain (~16 hours on RTX 3090)
python train.py --fresh

# 5. Stage 2: Chat pretrain (~3 hours)
python train_chat_pretrain.py

# 6. Stage 3: SFT (~10 minutes)
python train_sft.py --checkpoint checkpoints/chat_pretrain/latest.pt \
                    --data data/personal/sft_diverse_only.jsonl \
                    --epochs 5 --lr 3e-5

# 7. Chat!
python chat.py
```

## The Bug That Changed Everything

The original pretraining ran for 122k steps (~2 days, 4.4B tokens) but **never produced coherent text**.

**Root cause**: A double-shift label bug. The model's `forward()` shifts labels internally, but every training script ALSO pre-shifted — so the model learned to predict **2 tokens ahead** instead of 1.

```python
# BUG — every training script had this:
input_ids = tokens[:-1]   # shifted once in data prep
labels = tokens[1:]       # shifted again → model predicts 2-ahead!

# FIX — model.forward() handles the shift internally:
input_ids = tokens[:seq_len]
labels = tokens[:seq_len]  # same tensor, no pre-shifting
```

After fixing this, a fresh pretrain produces coherent English in ~6 hours.

## Key Lessons Learned

1. **SFT data must be diverse** — 6,010 tech-only conversations produced a model that could only talk about programming. Adding greetings, math, factual Q&A, and chitchat was essential.
2. **Chat pretraining bridges the gap** — Going straight from base pretrain to SFT doesn't work well. An intermediate stage where the model sees dialogue format as raw text teaches conversational structure.
3. **Don't mix old SFT data into final SFT** — When we combined 6,010 original + 2,129 diverse conversations, the old data (74%) drowned out the new. Diverse-only was dramatically better.
4. **SFT overfits fast** — Best results at epoch 2-3. By epoch 9+, the model memorizes ~15 response templates and retrieves the wrong ones.
5. **Math requires exhaustive examples** — The model memorizes number pairs, not arithmetic. Including ALL multiplication tables 2-12 was necessary for reliable math.

## Concept Learning Experiments

We ran experiments on what a 164M model can learn through SFT:

| Finding | Detail |
|---------|--------|
| Every concept trains to 100% | Yes/no, opposites, Q&A, math, chat — all learnable |
| Does it generalize? | No — 100% on training, 7-40% on held-out |
| Does it forget? | Catastrophically — 100% → 0% after sequential new tasks |
| Does replay fix forgetting? | Yes — 30% replay maintains 100% retention |
| Best strategy? | Joint training with experience replay |

```bash
python legacy/concept_experiments_v2.py    # SFT masking experiments
python legacy/concept_experiments_v3.py    # generalization/composition/retention
```

## Tournament Architecture Search

We trained **10 different architectures** in a 3-round elimination tournament to pick the best design:

| # | Variant | Params | Key Idea | Result |
|---|---------|--------|----------|--------|
| 1 | Dense small | 75.5M | Small baseline | Eliminated R2 |
| 2 | **Dense medium** | **163.6M** | **Standard GQA** | **WINNER** |
| 3 | DiffAttn small | 81.8M | Noise-canceling attention | Eliminated R2 |
| 4 | DiffAttn medium | 179.4M | Medium + differential | Eliminated R3 |
| 5 | MoE small | 185.7M (100M active) | Sparse routing | Eliminated R2 |
| 6 | MoE medium | 516M (~180M active) | Large sparse | Eliminated R3 |
| 7 | Hybrid small | 190.4M | DiffAttn + MoE | Eliminated R1 |
| 8 | Hybrid medium | 528.6M | Full hybrid | OOM |
| 9 | Deep narrow | 70.8M | 32 layers, narrow | Eliminated R3 |
| 10 | Wide shallow | 121.7M | 8 layers, wide | OOM |

**Winner: Dense medium (v02)** — Simple, efficient, and learned fastest with more training steps.

## Project Structure

```
chat_hamner/
├── model.py                  # Core transformer architecture (164M)
├── chat.py                   # Interactive CLI chat
├── train.py                  # Stage 1: Base pretraining (FineWeb-Edu)
├── train_chat_pretrain.py    # Stage 2: Chat pretraining (dialogue mix)
├── train_sft.py              # Stage 3: Supervised fine-tuning
├── generate_sft_v3.py        # Generate diverse SFT data (2,129 convos)
├── generate_sft_data.py      # Generate tech SFT data (6,010 convos)
├── test_ood.py               # Out-of-distribution evaluation
├── synthetic_tasks.py        # Synthetic task generators
├── plot_training.py          # Training metrics visualization
├── process_youtube.py        # YouTube transcript processor
├── data/
│   └── personal/             # SFT data, YouTube transcripts, voice samples
├── checkpoints/
│   ├── pretrain_v2/          # Stage 1 checkpoint
│   ├── chat_pretrain/        # Stage 2 checkpoint
│   └── sft/                  # Stage 3 checkpoints (best.pt = final model)
├── logs/                     # Training logs, metrics CSVs, sample generations
└── legacy/                   # Old experiments: tournament, concept experiments,
                              # alternative architectures (Mamba, RWKV, xLSTM, etc.)
```

## System Requirements

- NVIDIA GPU with 24GB+ VRAM (tested on RTX 3090)
- 32GB+ RAM
- Python 3.10+
- CUDA 12.x
- ~20GB disk space for checkpoints

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.

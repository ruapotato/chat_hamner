# Chat Hamner

A friendly conversational AI trained from scratch on a single RTX 3090.

Not a coding assistant. Not a knowledge oracle. Just a **friend** — warm, pragmatic, and genuinely interested in chatting.

## Current Status: Train V3.1 — 50M, 8-Stage Curriculum

A 50M model trained through **8-stage plateau-driven curriculum learning** with triple-confirmed stage advancement. Each stage adds a new data focus while keeping all prior sources active to prevent catastrophic forgetting. Stages advance only when loss plateaus across 3 consecutive checks — no fixed step counts, no flukes.

### Why V3.1?

V3's first run completed all 4 stages in just 1.5h / 16k steps / 0.4B tokens. The model produced grammatically correct text but had zero conversational coherence — it was a TinyStories engine with a thin veneer of tech words. Root causes: plateau threshold too loose (0.5%), single check triggered advancement, only 4 stages, and no dedicated reasoning/context stages.

V3.1 fixes all of this: stricter plateau detection (0.1% threshold, 3 consecutive strikes needed), 8 stages with dedicated reasoning and context phases, and 3 new synthetic task types (context recall, logic, comparison). Expected: ~200k-400k steps, 5-10B tokens, ~30-60 hours at ~71k tok/s.

| # | Stage | Focus | Primary Data | Synth Diff | LR |
|---|-------|-------|-------------|-----------|-----|
| 1 | Structure | English grammar | 70% TinyStories, 30% synthetic | 1 | 3e-4 |
| 2 | Reasoning | Logic/math/patterns | 55% synthetic, 35% TinyStories, 10% FineWeb | 3 | 2.5e-4 |
| 3 | Knowledge | World knowledge | 60% FineWeb, 20% TinyStories, 20% synthetic | 3 | 2e-4 |
| 4 | Context | Fact tracking/recall | 35% FineWeb, 35% synthetic, 20% TinyStories, 10% SFT | 4 | 1.5e-4 |
| 5 | Deep Knowledge | Consolidate reasoning+knowledge | 40% FineWeb, 25% synthetic, 15% TinyStories, 15% SFT, 5% personal | 4 | 1.2e-4 |
| 6 | Dialogue | Conversational format | 40% SFT, 20% FineWeb, 15% TinyStories, 15% synthetic, 10% personal | 4 | 8e-5 |
| 7 | Synthesis | Integrate everything | 25% FineWeb, 25% SFT, 20% synthetic, 15% TinyStories, 15% personal | 5 | 5e-5 |
| 8 | Voice | Personality | 35% personal, 25% SFT, 15% FineWeb, 10% TinyStories, 15% synthetic | 5 | 3e-5 |

### V2 Results (for reference)

| Metric | v1 (buggy, 122k steps) | v2 (fixed, 106k steps) |
|--------|----------------------|------------------------|
| Label handling | Predicted 2-ahead (broken) | Predicted 1-ahead (correct) |
| Loss | 4.54 (never improved) | 3.08 (plateaued) |
| Coherent output? | Never | Partially, but still incoherent |
| Tokens processed | 4.4B | 2.6B |

## Architecture

Hamner is a transformer built from scratch using modern components:

- **RMSNorm** — Faster, more stable normalization
- **SwiGLU** — Gated activation for better gradient flow
- **RoPE** — Rotary positional embeddings for position encoding
- **GQA** — Grouped query attention for memory efficiency
- **Mixed precision (fp16)** — 2x memory savings
- **torch.compile** — PyTorch 2.x compilation for faster training
- **Gradient checkpointing** — Trades compute for VRAM savings

The architecture was selected through a tournament of 10 competing designs (see [Tournament](#tournament-architecture-search) below).

### Model Specs (V3 — 50M)

| Parameter | Value |
|-----------|-------|
| Hidden size | 512 |
| Layers | 8 |
| Attention heads | 8 (2 KV, GQA 4:1) |
| MLP type | Dense SwiGLU |
| Intermediate size | 1365 |
| Parameters | ~47M |
| Sequence length | 1024 |
| Tokenizer | cosmo2-tokenizer (49,152 vocab) |

### Previous Model (V2 — 164M)

| Parameter | Value |
|-----------|-------|
| Hidden size | 768 |
| Layers | 20 |
| Attention heads | 12 (4 KV) |
| MLP type | Dense SwiGLU |
| Intermediate size | 2048 |
| Parameters | 163.6M |

## The Bug That Changed Everything

The original pretraining ran for 122k steps (~2 days, 4.4B tokens) but **never produced coherent text**. The model's output was garbled nonsense despite steadily decreasing loss.

**Root cause**: A double-shift label bug existed in every training script since day one.

The model's `forward()` method shifts labels internally:
```python
# model.py line 364 — model handles the shift
loss = CE(logits[..., :-1, :], labels[..., 1:])
```

But every training script ALSO pre-shifted:
```python
# ALL training scripts had this bug:
input_ids = tokens[:-1]   # shifted once
labels = tokens[1:]       # shifted twice!
```

**Result**: The model learned to predict **2 tokens ahead** instead of 1. Every gradient update was teaching the wrong objective. The loss went down because predicting 2-ahead is still a learnable task, but the model could never produce coherent next-token predictions.

**Fix**: Pass the same unshifted tokens as both `input_ids` and `labels`. The model handles the shift internally.

```python
# CORRECT — model.forward() shifts labels internally
input_ids = tokens[:seq_len]
labels = tokens[:seq_len]  # same! no pre-shifting
```

After fixing this, a fresh pretrain produces coherent English in ~6 hours on a single 3090.

## Concept Learning Experiments

After discovering the bug fix, we ran extensive experiments on what a 164M model can learn through SFT (supervised fine-tuning). Key findings:

### What Works (v2 experiments)

After the bug fix, **every concept type trains to 100%**:

| Concept | Accuracy | Example |
|---------|----------|---------|
| Yes/No classification | 100% | "Is 7 bigger than 3?" → "Yes" |
| Opposites | 100% | "The opposite of hot is" → "cold" |
| Q&A facts | 100% | "Q: What color is the sky?" → "Blue" |
| Number echoing | 100% | "Echo 42 =" → "42" |
| Number words | 100% | "13 in words is" → "thirteen" |
| Addition | 100% | "5 + 7 =" → "12" |
| Chat responses | 100% | "User: Hello!" → "Hi there!" |
| Multi-task (all 7) | 97-100% | All concepts simultaneously |

### The Hard Questions (v3 experiments)

| Question | Finding |
|----------|---------|
| **Does it generalize?** | No — 100% on training examples, 7-40% on held-out. It memorizes, not learns. |
| **Do skills compose?** | Partially — learning "opposites" + "Q&A" separately gives 10-45% on "Q&A about opposites" depending on training approach. Direct training: 100%. |
| **Does it forget?** | Yes, catastrophically — QA accuracy goes 100% → 10% → 0% after sequential training on new tasks. |
| **Does replay fix forgetting?** | Yes, completely — 30% replay of old data during new task training maintains 100% retention. |
| **How many facts can it hold?** | 50+ easily. 164M params has plenty of memorization capacity. |
| **Best training strategy?** | Joint training with experience replay. Never train sequentially. |

### Key Insight

SFT alone can't build a chatbot — it only creates lookup tables. The model needs a good pretrained base with real language understanding before SFT will generalize to unseen inputs. **That's why the corrected pretraining is the most important step.**

## Training

### Data

- **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** — Simple narrative English (stage 1 primary)
- **[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)** — 10B token educational web text (stage 2 primary)
- **Synthetic tasks** — 9 task types: arithmetic, counting, sorting, brackets, listops, copy/repeat, context recall, logic, comparison (all stages, difficulty scales 1-5)
- **SFT conversations** — Chat-formatted dialogue data (stage 3 primary)
- **Personal voice** — YouTube transcriptions from [@davidhamner](https://www.youtube.com/@davidhamner) (stage 4 primary)

### Train V3.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 24 |
| Sequence length | 1024 |
| Learning rate | Per-stage: 3e-4 → 2.5e-4 → 2e-4 → 1.5e-4 → 1.2e-4 → 8e-5 → 5e-5 → 3e-5 |
| LR schedule | 300-step warmup then constant (per stage) |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, wd=0.1), reset each stage |
| Gradient clipping | 1.0 |
| Precision | fp16 mixed precision |
| Gradient checkpointing | Enabled |
| torch.compile | Enabled |
| Plateau window | 1000 recent / 2000 compare |
| Plateau threshold | 0.1% improvement (triple-confirmed, 3 consecutive strikes) |
| Plateau min steps | 5000 per stage before checking |
| Expected throughput | ~71k tok/sec (~6.1B/day) |
| Hardware | Single RTX 3090 (24GB) |

### Training Progress

Run `python plot_training.py --save` to generate latest plots.

## Usage

### Train

```bash
# Set up environment
cd chat_hamner
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train V3 — 50M staged learning (recommended)
python train_v3.py                    # start fresh
python train_v3.py --resume           # resume from latest checkpoint
python train_v3.py --stage 2          # skip to stage N (0-indexed)

# Legacy: pretrain V2 (164M, FineWeb-only)
python train.py
python train.py --fresh
python train.py --checkpoint path/to/checkpoint.pt
```

### Plot Training Metrics

```bash
python plot_training.py --save           # save plots to logs/plots/
python plot_training.py --live           # auto-refresh every 60s
python plot_training.py --dashboard      # single dashboard image
```

### Run Concept Experiments

```bash
python concept_experiments_v2.py         # SFT masking experiments
python concept_experiments_v3.py         # generalization/composition/retention
```

### Process YouTube Data

```bash
# Download transcriptions (requires yt-dlp)
yt-dlp --write-auto-sub --sub-lang en --skip-download -o "data/youtube/%(id)s_%(title)s" "https://www.youtube.com/@davidhamner"

# Process into training samples
python process_youtube.py
```

## Tournament Architecture Search

Rather than picking one architecture, we trained **10 different variants** in a 3-round elimination tournament.

### The 10 Contestants

| # | Variant | Params | Attention | MLP | Key Idea |
|---|---------|--------|-----------|-----|----------|
| 1 | v01_dense_small | 75.5M | Standard GQA | Dense | Small baseline |
| 2 | **v02_dense_medium** | **163.6M** | **Standard GQA** | **Dense** | **WINNER** |
| 3 | v03_diffattn_small | 81.8M | Differential | Dense | Noise-canceling attention |
| 4 | v04_diffattn_medium | 179.4M | Differential | Dense | Medium + diff attention |
| 5 | v05_moe_small | 185.7M (100M active) | Standard GQA | MoE-8 top-2 | Sparse routing |
| 6 | v06_moe_medium | 516M (~180M active) | Standard GQA | MoE-8 top-2 | Large sparse model |
| 7 | v07_hybrid_small | 190.4M | Differential | MoE-8 top-2 | DiffAttn + MoE combo |
| 8 | v08_hybrid_medium | 528.6M | Differential | MoE-8 top-2 | Full hybrid (OOM) |
| 9 | v09_deep_narrow | 70.8M | Standard GQA | Dense | 32 layers, narrow |
| 10 | v10_wide_shallow | 121.7M | Standard GQA | Dense | 8 layers, wide (OOM) |

**Winner: v02_dense_medium** — Simple, efficient, and learned fastest with more training steps. Differential attention helped early but standard GQA caught up by Round 3.

## Project Structure

```
chat_hamner/
├── model.py                  # Core architecture (supports all model sizes)
├── train_v3.py               # Train V3 — 50M staged learning (CURRENT)
├── train.py                  # Pretraining v2 — 164M FineWeb-only (legacy)
├── train_sft.py              # SFT fine-tuning pipeline
├── chat.py                   # Interactive CLI chat
├── synthetic_tasks.py        # 9 synthetic task generators (arithmetic, logic, context recall, etc.)
├── tournament.py             # Tournament architecture search
├── variants.py               # 10 variant configs + emotional layer groups
├── curriculum_train.py       # Curriculum training (legacy, reference)
├── concept_experiments.py    # SFT concept experiments
├── process_youtube.py        # YouTube transcript processor
├── plot_training.py          # Training metrics visualization
├── data/
│   └── personal/             # YouTube transcripts, SFT convos, training samples
├── checkpoints/
│   ├── v3/                   # Train V3 checkpoints (current)
│   ├── pretrain_v2/          # V2 pretraining (164M, plateaued)
│   └── training/             # Legacy v1 pretraining (buggy labels)
└── logs/
    ├── v3.log                # V3 training log
    ├── v3_metrics.csv        # V3 metrics (CSV)
    ├── v3_samples.jsonl      # V3 sample generations
    ├── pretrain_v2.log       # V2 training log
    └── plots/                # Generated plots
```

### Legacy Files

- `train.py` — V2 pretraining (164M, plateaued at loss 3.08)
- `curriculum_train.py` — Curriculum training on buggy v1 base
- `checkpoints/training/` — v1 pretraining (122k steps, double-shift bug)
- `checkpoints/curriculum/` — Curriculum training checkpoints
- `checkpoints/final/`, `checkpoints/round1-3/` — Tournament checkpoints

## System Requirements

- NVIDIA GPU with 24GB+ VRAM (tested on RTX 3090)
- 32GB+ RAM
- Python 3.10+
- CUDA 12.x
- ~50GB disk space for checkpoints

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.

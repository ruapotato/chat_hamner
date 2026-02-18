# Chat Hamner

A friendly conversational AI trained from scratch on a single RTX 3090.

Not a coding assistant. Not a knowledge oracle. Just a **friend** — warm, pragmatic, and genuinely interested in chatting.

## Current Status: Pretraining v2 (RUNNING)

Fresh pretraining with **correct label handling**, started 2026-02-18.
Previous 122k-step run was trained with a critical double-shift label bug (see [The Bug](#the-bug-that-changed-everything) below).

| Metric | v1 (buggy, 122k steps) | v2 (fixed, in progress) |
|--------|----------------------|------------------------|
| Label handling | Predicted 2-ahead (broken) | Predicted 1-ahead (correct) |
| Training time | ~2 days | ~6 hours so far |
| Loss | 4.54 (never improved) | **~3.4 and dropping** |
| Coherent output? | Never | **Yes, within hours** |
| Tokens processed | 4.4B | ~0.6B (and counting) |
| Checkpoint | `checkpoints/training/latest.pt` (legacy) | `checkpoints/pretrain_v2/latest.pt` |

### Sample Output (step ~23k, 0.6B tokens, 6 hours)

> **Prompt**: "Once upon a time there was a"
> **Output**: "time when the people did not have any money to buy their products. We all know that you don't have much money at all. We need to get up and go about this."

> **Prompt**: "The meaning of life is"
> **Output**: "the most important thing in the world. It is the most basic of all the things that we can do."

This is from scratch on a single 3090 — random weights to coherent English in 6 hours.

## Architecture

Hamner is a 164M parameter transformer built from scratch using modern components:

- **RMSNorm** — Faster, more stable normalization
- **SwiGLU** — Gated activation for better gradient flow
- **RoPE** — Rotary positional embeddings for position encoding
- **GQA** — Grouped query attention (12 heads, 4 KV heads) for memory efficiency
- **Mixed precision (fp16)** — 2x memory savings
- **torch.compile** — PyTorch 2.x compilation for faster training

The winner was selected through a tournament of 10 competing architectures (see [Tournament](#tournament-architecture-search) below).

### Model Specs

| Parameter | Value |
|-----------|-------|
| Hidden size | 768 |
| Layers | 20 |
| Attention heads | 12 (4 KV) |
| MLP type | Dense SwiGLU |
| Intermediate size | 2048 |
| Parameters | 163.6M |
| Sequence length | 1024 |
| Tokenizer | cosmo2-tokenizer (49,152 vocab) |

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

- **Pre-training**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (10B token sample, streamed)
- **Personal voice**: YouTube transcriptions from [@davidhamner](https://www.youtube.com/@davidhamner) (1,873 samples, mixed in at 5%)

### Pretraining v2 (Current)

| Parameter | Value |
|-----------|-------|
| Batch size | 24 |
| Sequence length | 1024 |
| Learning rate | 2e-4 (linear warmup 2k steps + cosine decay) |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, wd=0.1) |
| Gradient clipping | 1.0 |
| Precision | fp16 mixed precision |
| Gradient checkpointing | Enabled |
| torch.compile | Enabled |
| Target | 400k steps / ~10B tokens |
| Throughput | ~26k tokens/sec |
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

# Start training (auto-resumes from latest checkpoint)
python train.py

# Or start fresh
python train.py --fresh

# Resume from specific checkpoint
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
├── model.py                  # Core architecture (164M param transformer)
├── train.py                  # Pretraining v2 (RUNNING — correct labels)
├── tournament.py             # Tournament architecture search
├── variants.py               # 10 variant configs + emotional layer groups
├── concept_experiments_v2.py  # SFT concept experiments (bug fix validation)
├── concept_experiments_v3.py  # Generalization/composition/retention tests
├── process_youtube.py        # YouTube transcript processor
├── plot_training.py          # Training metrics visualization
├── data/
│   └── personal/             # YouTube transcripts + training samples
├── checkpoints/
│   ├── pretrain_v2/          # Current pretraining (correct labels)
│   └── training/             # Legacy v1 pretraining (buggy labels)
└── logs/
    ├── pretrain_v2.log       # Current training log
    ├── pretrain_v2_metrics.csv
    ├── experiments_v2.log    # SFT experiment results
    ├── experiments_v3.log    # Scaffolding experiment results
    └── plots/                # Generated plots
```

### Legacy Files

These are from the pre-bug-fix era and kept for reference:

- `checkpoints/training/` — v1 pretraining (122k steps, double-shift bug)
- `checkpoints/curriculum/` — Curriculum training on buggy base
- `checkpoints/final/`, `checkpoints/round1-3/` — Tournament checkpoints
- `curriculum_train.py`, `baby_train.py`, `synthetic_tasks.py` — Curriculum pipeline (superseded by correct pretraining)
- `logs/metrics.csv`, `logs/training.log` — v1 training logs

## System Requirements

- NVIDIA GPU with 24GB+ VRAM (tested on RTX 3090)
- 32GB+ RAM
- Python 3.10+
- CUDA 12.x
- ~50GB disk space for checkpoints

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.

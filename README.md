# Chat Hamner

A conversational AI trained from scratch on a single RTX 3090. 164M parameters, custom transformer architecture, no pretrained weights.

Not a coding assistant. Not a knowledge oracle. Just **Al Hamner** — sharp-witted, opinionated, and built from zero.

## Sample Conversation

From an earlier V2 model trained on only 2.6B tokens in ~1 day. V4 (currently training on 14B tokens over ~7 days) should be significantly better.

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

## Training Pipeline (V4)

Inspired by [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): massive overtraining on diverse data with a multi-stage curriculum. 5 stages, ~7 days on a single RTX 3090.

### Stage 1: Base Pretraining (~4.5 days)

Train the base language model on diverse web text from two high-quality sources.

```bash
python train_pretrain.py --fresh
```

- **Data**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 60% + [DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) 40% (streamed)
- **Steps**: 400,000 (~10B tokens)
- **LR**: 2e-4 with Warmup-Stable-Decay (WSD) schedule
- **Batch size**: 24 x 1024 tokens
- **Output**: `checkpoints/pretrain_v4/latest.pt`

### Stage 2: Code + Math Annealing (~32 hours)

Anneal with code and math data mixed in, linearly decaying LR to zero.

```bash
python train_pretrain.py --stage anneal
```

- **Base**: Stage 1 checkpoint
- **Data**: FineWeb-Edu 40% + DCLM 20% + [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) 25% + [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) 15%
- **Steps**: 122,000 (~3B tokens)
- **Output**: `checkpoints/pretrain_v4_anneal/latest.pt`

### Stage 3: Chat Pretraining (~11 hours)

Continue on a dialogue-heavy mix so the model learns conversational format as natural language.

```bash
python train_chat_pretrain.py
```

- **Base**: Stage 2 checkpoint
- **Data mix**: [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) 40% + FineWeb 25% + DCLM 15% + personal voice 10% + synthetic tasks 10%
- **Steps**: 40,000 (~1B tokens)
- **Output**: `checkpoints/chat_pretrain/latest.pt`

### Stage 4: Supervised Fine-Tuning (~3 hours)

Fine-tune on ~104k conversations with loss computed only on assistant response tokens. Custom data upweighted 2x.

```bash
python prepare_sft_data.py   # downloads SmolTalk, combines with custom data
python train_sft.py
```

- **Base**: Stage 3 checkpoint
- **Data**: ~104k conversations (100k SmolTalk + 2,129 custom diverse + 2,000 custom tech)
- **Epochs**: 3 with early stopping (patience=3)
- **LR**: 1e-4 with linear warmup
- **Output**: `checkpoints/sft/best.pt`

### Stage 5: DPO Alignment (~1-2 hours)

Direct Preference Optimization using human preference data.

```bash
python train_dpo.py
```

- **Base**: Stage 4 checkpoint
- **Data**: [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) (~60k preference pairs)
- **Epochs**: 2, beta=0.1, LR=1e-6
- **Output**: `checkpoints/dpo/best.pt`

## Training Data

- **[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)** — Educational web text (base pretraining, chat pretrain)
- **[DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)** — General web text (base pretraining, annealing, chat pretrain)
- **[StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata)** — Code (annealing stage)
- **[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)** — Math reasoning (annealing stage)
- **[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)** — 100k high-quality conversations (chat pretrain + SFT)
- **[UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)** — Preference pairs (DPO alignment)
- **Custom diverse SFT** — 2,129 conversations: greetings, identity, math, opinions, reasoning
- **Custom tech SFT** — 6,010 tech-focused conversations (2,000 sampled for SFT)
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

# 3. Prepare SFT data (downloads SmolTalk, combines with custom data)
python prepare_sft_data.py

# 4. Stage 1: Base pretrain (~4.5 days on RTX 3090)
python train_pretrain.py --fresh

# 5. Stage 2: Code + Math anneal (~32 hours)
python train_pretrain.py --stage anneal

# 6. Stage 3: Chat pretrain (~11 hours)
python train_chat_pretrain.py

# 7. Stage 4: SFT (~3 hours)
python train_sft.py

# 8. Stage 5: DPO alignment (~1-2 hours)
python train_dpo.py

# 9. Chat!
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
├── train_pretrain.py         # Stages 1-2: Base pretraining + annealing
├── train_chat_pretrain.py    # Stage 3: Chat pretraining (dialogue mix)
├── train_sft.py              # Stage 4: Supervised fine-tuning
├── train_dpo.py              # Stage 5: DPO alignment
├── prepare_sft_data.py       # Download/prepare SFT data (~104k convos)
├── generate_sft_v3.py        # Generate diverse SFT data (2,129 convos)
├── generate_sft_data.py      # Generate tech SFT data (6,010 convos)
├── test_ood.py               # Out-of-distribution evaluation
├── synthetic_tasks.py        # Synthetic task generators
├── plot_training.py          # Training metrics visualization (--v4 for dashboard)
├── process_youtube.py        # YouTube transcript processor
├── data/
│   └── personal/             # SFT data, YouTube transcripts, voice samples
├── checkpoints/
│   ├── pretrain_v4/          # Stage 1 checkpoint
│   ├── pretrain_v4_anneal/   # Stage 2 checkpoint
│   ├── chat_pretrain/        # Stage 3 checkpoint
│   ├── sft/                  # Stage 4 checkpoints (best.pt)
│   └── dpo/                  # Stage 5 checkpoints (best.pt = final model)
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

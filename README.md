# Chat Hamner

A friendly conversational AI trained from scratch on a single RTX 3090.

Not a coding assistant. Not a knowledge oracle. Just a **friend** — warm, pragmatic, and genuinely interested in chatting.

## Architecture

Hamner is a 164M parameter transformer built from scratch using modern components:

- **RMSNorm** — Faster, more stable normalization
- **SwiGLU** — Gated activation for better gradient flow
- **RoPE** — Rotary positional embeddings for position encoding
- **GQA** — Grouped query attention (12 heads, 4 KV heads) for memory efficiency
- **Mixed precision (fp16)** — 2x memory savings
- **Gradient checkpointing** — Trade compute for memory

The winner was selected through a tournament of 10 competing architectures (see below).

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

## Tournament Architecture Search

Rather than picking one architecture, we trained **10 different variants** in a 3-round elimination tournament. Each variant got a fair shot, and the data decided the winner.

### The 10 Contestants

| # | Variant | Params | Attention | MLP | Key Idea |
|---|---------|--------|-----------|-----|----------|
| 1 | v01_dense_small | 75.5M | Standard GQA | Dense | Small baseline |
| 2 | **v02_dense_medium** | **163.6M** | **Standard GQA** | **Dense** | **Medium baseline (WINNER)** |
| 3 | v03_diffattn_small | 81.8M | Differential | Dense | Noise-canceling attention |
| 4 | v04_diffattn_medium | 179.4M | Differential | Dense | Medium + diff attention |
| 5 | v05_moe_small | 185.7M (100M active) | Standard GQA | MoE-8 top-2 | Sparse routing |
| 6 | v06_moe_medium | 516M (~180M active) | Standard GQA | MoE-8 top-2 | Large sparse model |
| 7 | v07_hybrid_small | 190.4M | Differential | MoE-8 top-2 | DiffAttn + MoE combo |
| 8 | v08_hybrid_medium | 528.6M | Differential | MoE-8 top-2 | Full hybrid (OOM) |
| 9 | v09_deep_narrow | 70.8M | Standard GQA | Dense | 32 layers, narrow |
| 10 | v10_wide_shallow | 121.7M | Standard GQA | Dense | 8 layers, wide (OOM) |

### Round 1 — All 10 Variants (500 steps each)

| Rank | Variant | Loss | Notes |
|------|---------|------|-------|
| 1 | v04_diffattn_medium | 7.095 | Differential attention led early |
| 2 | v02_dense_medium | 7.111 | Close second |
| 3 | v03_diffattn_small | 7.154 | Small diff attn held up |
| 4 | v01_dense_small | 7.164 | Small but competitive |
| 5 | v07_hybrid_small | 7.241 | Advanced to Round 2 |
| 6 | v09_deep_narrow | 7.269 | Eliminated — too deep for this scale |
| 7 | v05_moe_small | 7.287 | Eliminated — routing not converged |
| 8 | v06_moe_medium | 7.287 | Eliminated — routing not converged |
| 9 | v08_hybrid_medium | OOM | Eliminated — too large for 24GB |
| 10 | v10_wide_shallow | OOM | Eliminated — too large for 24GB |

### Round 2 — Top 5 (1,000 more steps)

| Rank | Variant | Loss | Delta from R1 |
|------|---------|------|---------------|
| 1 | v04_diffattn_medium | 6.846 | -0.249 |
| 2 | v02_dense_medium | 6.867 | -0.244 |
| 3 | v03_diffattn_small | 6.931 | -0.223 |
| 4 | v01_dense_small | 6.944 | Eliminated |
| 5 | v07_hybrid_small | 6.996 | Eliminated |

### Round 3 — Final 3 (2,000 more steps)

| Rank | Variant | Loss | Delta from R2 |
|------|---------|------|---------------|
| 1 | **v02_dense_medium** | **6.706** | **-0.161** |
| 2 | v04_diffattn_medium | 6.712 | -0.134 |
| 3 | v03_diffattn_small | 6.804 | -0.127 |

**Winner: v02_dense_medium** — The standard GQA + dense SwiGLU architecture at 164M parameters. Simple, efficient, and it learned fastest with more training steps.

### What We Learned

1. **Differential attention helps early but standard catches up.** v04 led through Rounds 1-2, but v02 overtook it in Round 3 as it continued learning efficiently.
2. **Medium models beat small models.** More parameters = faster loss reduction, within VRAM budget.
3. **MoE underperforms at short horizons.** Expert routing needs thousands of steps to converge — at 500 steps the MoE models couldn't compete.
4. **Deep and narrow is worse than moderate depth.** 32 layers at hidden=384 underperformed 20 layers at hidden=768.
5. **Simple architectures scale better on a single GPU.** Dense models have lower overhead and more efficient memory usage than MoE on single-GPU training.

## Training

### Data

- **Pre-training**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (10B token sample, streamed from HuggingFace)
- **Personal voice**: YouTube transcriptions from [@davidhamner](https://www.youtube.com/@davidhamner) (93 videos, 65K+ words, mixed in at 5%)

The personal data is mixed into every training batch so the model learns David's conversational style alongside general language ability.

### Training Progress

Training on RTX 3090 at ~12,000 tokens/sec:

| Milestone | Steps | Tokens | Loss | Perplexity |
|-----------|-------|--------|------|------------|
| Tournament start | 0 | 0 | ~10.9 | ~54,000 |
| Tournament end | 3,500 | ~57M | ~6.7 | ~812 |
| Extended training | 30,500 | ~500M | ~5.5 | ~245 |
| Target | 300,000 | ~4.9B | TBD | TBD |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 16 |
| Sequence length | 1024 |
| Learning rate | 3e-4 |
| LR schedule | Linear warmup (2K steps) + cosine decay |
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| Gradient clipping | 1.0 |
| Precision | fp16 mixed precision |

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

### Chat

```bash
python chat.py --checkpoint checkpoints/training/latest.pt
```

### Process YouTube Data

```bash
# Download transcriptions (requires yt-dlp)
yt-dlp --write-auto-sub --sub-lang en --skip-download -o "data/youtube/%(id)s_%(title)s" "https://www.youtube.com/@davidhamner"

# Process into training samples
python process_youtube.py
```

### Run the Tournament

```bash
# Train all 10 variants and pick the best
python tournament.py
```

## System Requirements

- NVIDIA GPU with 24GB+ VRAM (tested on RTX 3090)
- 32GB+ RAM
- Python 3.10+
- CUDA 12.x
- ~50GB disk space for checkpoints

## Project Structure

```
chat_hamner/
├── model.py              # Core architecture (attention, MLP, transformer blocks)
├── train.py              # Production training script with streaming + auto-resume
├── tournament.py         # Tournament architecture search system
├── variants.py           # 10 architecture variant configurations
├── chat.py               # Interactive chat interface
├── process_youtube.py    # YouTube VTT transcript processor
├── prepare_data.py       # Data preparation utilities
├── data/
│   ├── personal/         # YouTube transcripts + training samples
│   └── youtube/          # Raw VTT subtitle files
├── checkpoints/          # Model checkpoints (not in git)
└── logs/                 # Training logs
```

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.

Built by David Hamner with help from Claude.

# Hamner Scaling Analysis
**Date: 2026-02-17 | Step 122k | Loss 4.42 | 4.4B tokens**

## Problem: Model is Hitting Capacity Wall

The current 164M param model is projected to reach loss 4.18 / PPL 66 at
the end of training (14.7B tokens). This is above the coherence threshold
(~3.5 loss / ~33 PPL). The model will not produce readable text from
pre-training alone.

**Evidence:**
- Scaling slope: -0.062 (very shallow — diminishing returns)
- Chinchilla optimal for 164M params: 3.3B tokens. We're already at 4.4B
  and heading to 14.7B — that's 4.5x overtrained on data.
- Sample generations at step 122k are still incoherent fragments
- Loss dropped only 0.22 over the last 32k steps (1.6B tokens)

**Conclusion:** More data won't fix this. The bottleneck is model capacity.

## Hardware Budget

- **GPU:** NVIDIA RTX 3090, 24GB VRAM
- **Current usage:** ~22GB (92%) with batch=48, seq=1024, no grad checkpointing
- **Available with grad checkpointing:** Re-enabling gradient checkpointing
  trades ~30% throughput for ~60% less activation memory. This is the key
  lever for fitting a bigger model.

## Scaling Options

### Option A: Scale Wide (Recommended)
**321M params — 2x current**

```
hidden_size       = 1024    (was 768)
num_layers        = 24      (was 20)
num_attention_heads = 16    (was 12)
num_kv_heads      = 4       (unchanged, GQA)
expert_intermediate_size = 2816  (was 2048, keeping ~2.75x ratio)
max_seq_len       = 1024    (unchanged)
gradient_checkpointing = True   (was False)
use_differential_attention = False
```

| Metric | Current (164M) | Option A (321M) |
|--------|---------------|-----------------|
| Params | 164M | 321M |
| Batch size | 48 | 24 |
| Seq len | 1024 | 1024 |
| Grad checkpointing | Off | On |
| Est. VRAM | ~22GB | ~16GB |
| Tok/step | 49,152 | 24,576 |
| Chinchilla optimal | 3.3B tok | 6.4B tok |
| Training target | 14.7B tok | ~10B tok |
| Expected final loss | 4.18 | ~3.3-3.5 |
| Expected final PPL | 66 | ~27-33 |

**Why this works:** Doubling params roughly halves the loss gap to the
irreducible minimum (empirical scaling law). A 321M model trained to 10B
tokens should land solidly in the coherent zone. The batch size drop (48->24)
means ~half the tokens per step, but we need fewer total tokens, so wall-clock
time is comparable.

### Option A-variant: Pre-train Short, SFT Long
Same 321M architecture, but pre-train at seq_len=512:

- batch=32 fits comfortably (est. ~9GB)
- 50% more tokens per step vs seq=1024
- Pre-train to ~10B tokens, then extend to seq=1024 during SFT
- RoPE handles context extension natively
- Risk: slight quality loss from shorter pre-training context

### Option B: Scale Deep (260M)
Keep width at 768, increase layers from 20 to 36.

- 1.6x current params, simpler change (just num_layers)
- With grad checkpointing + batch=24: ~14GB
- Less impactful than width scaling at this model size
- Chinchilla optimal: 5.3B tokens

### Option C: MoE (280M active, 352M total)
4 experts, 2 active, expert_intermediate_size=1536.

- More total knowledge capacity at similar compute per token
- But MoE at small scale is finicky (expert collapse, routing instability)
- Added code complexity for modest gains
- Not recommended at this stage

## Recommendation

**Go with Option A (321M wide).** It's the straightforward path:
1. ~2x the model = enough capacity to reach coherent text
2. Grad checkpointing fits it easily on the 3090
3. ~10B tokens of training (Chinchilla-aligned)
4. Expected loss ~3.3-3.5 — solidly in the coherent zone
5. Then SFT on conversational data to unlock the chatbot

### Training Plan
```
Phase 1: Pre-train 321M model
  - ~10B tokens on FineWeb-Edu + personal data mix
  - batch=24, seq=1024, grad_checkpointing=True
  - LR: 2e-4, cosine decay to 2e-5
  - Steps: ~400K (10B / 24 / 1024)
  - Estimated wall-clock: ~6-7 days at ~20K tok/s

Phase 2: SFT
  - Fine-tune on data/sft/ conversational data
  - Lower LR (1e-5), seq=1024, small batch
  - Should produce coherent chatbot responses
```

### What We Lose
- Cannot continue from the current checkpoint (architecture change)
- ~16 hours of training on the 164M model

### What We Gain
- A model that will actually produce coherent text
- Better foundation for SFT
- Properly scaled for our hardware

"""
10 Parallel Architecture Variants for Tournament Training (v3)
==============================================================
All transformer-based (fully parallel, no sequential scans).
They compete on validation loss to find what works best.

All ~300-340M params, gradient checkpointing enabled.

 1. v01_transformer    - Standard GQA Transformer (baseline)
 2. v02_diff_attn      - Differential Attention (noise-canceling)
 3. v03_moe_8x         - 8-expert MoE, top-2 routing
 4. v04_diff_moe       - Diff Attention + MoE combined
 5. v05_deep_narrow    - Deep & narrow (hidden=768, 44 layers)
 6. v06_wide_shallow   - Wide & shallow (hidden=1536, 10 layers)
 7. v07_many_heads     - 32 attention heads (finer-grained)
 8. v08_emotional      - Emotional Transformer (slow middle layers)
 9. v09_mqa            - Multi-Query Attention (kv_heads=1)
10. v10_big_moe        - 16-expert MoE, top-2 routing
"""

from dataclasses import dataclass
from typing import Callable, Optional
from model import HamnerConfig


@dataclass
class VariantInfo:
    """Everything the tournament needs to know about a variant."""
    config: HamnerConfig
    description: str
    param_group_fn: Optional[Callable] = None  # fn(model, base_lr) -> param_groups


# ---------------------------------------------------------------------------
# Emotional variant: separate LR for middle layers
# ---------------------------------------------------------------------------

def emotional_param_groups(model, base_lr):
    """Create param groups with slower LR for emotional (middle) layers."""
    config = model.config
    num_layers = config.num_layers
    num_emotional = config.emotional_layers
    scale = config.emotional_lr_scale

    start = (num_layers - num_emotional) // 2
    end = start + num_emotional
    emotional_range = set(range(start, end))

    fast_params, slow_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_emotional = False
        if "layers." in name:
            try:
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                is_emotional = layer_idx in emotional_range
            except (ValueError, IndexError):
                pass
        (slow_params if is_emotional else fast_params).append(param)

    return [
        {"params": fast_params, "lr": base_lr},
        {"params": slow_params, "lr": base_lr * scale},
    ]


# ---------------------------------------------------------------------------
# Shared base kwargs
# ---------------------------------------------------------------------------

_BASE = dict(
    hidden_size=1024,
    num_attention_heads=16,
    num_kv_heads=4,
    num_experts=1,
    num_active_experts=1,
    expert_intermediate_size=2816,
    use_differential_attention=False,
    gradient_checkpointing=True,
    architecture="transformer",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_variant_configs() -> dict:
    """Return all 10 tournament variant definitions."""
    variants = {}

    # 1. Standard Transformer (~321M) - GQA baseline
    variants["v01_transformer"] = VariantInfo(
        config=HamnerConfig(**_BASE, num_layers=24),
        description="Standard Transformer (GQA, SwiGLU, RoPE)",
    )

    # 2. Differential Attention (~340M) - noise-canceling dual softmax
    variants["v02_diff_attn"] = VariantInfo(
        config=HamnerConfig(**{**_BASE, "use_differential_attention": True},
                            num_layers=20),
        description="Differential Attention (noise-canceling, 2x QK)",
    )

    # 3. MoE 8-expert (~330M total, ~120M active) - sparse routing
    variants["v03_moe_8x"] = VariantInfo(
        config=HamnerConfig(**{**_BASE, "num_experts": 8, "num_active_experts": 2,
                               "expert_intermediate_size": 1024},
                            num_layers=10),
        description="MoE 8-expert top-2 (sparse, more capacity)",
    )

    # 4. Diff Attention + MoE (~330M) - both innovations
    variants["v04_diff_moe"] = VariantInfo(
        config=HamnerConfig(**{**_BASE, "use_differential_attention": True,
                               "num_experts": 8, "num_active_experts": 2,
                               "expert_intermediate_size": 1024},
                            num_layers=8),
        description="Diff Attention + MoE (dual innovation)",
    )

    # 5. Deep & Narrow (~319M) - hidden=768, many layers
    variants["v05_deep_narrow"] = VariantInfo(
        config=HamnerConfig(**{**_BASE, "hidden_size": 768,
                               "num_attention_heads": 12, "num_kv_heads": 4,
                               "expert_intermediate_size": 2112},
                            num_layers=44),
        description="Deep & Narrow (768h x 44L, more depth)",
    )

    # 6. Wide & Shallow (~325M) - hidden=1536, few layers
    variants["v06_wide_shallow"] = VariantInfo(
        config=HamnerConfig(**{**_BASE, "hidden_size": 1536,
                               "num_attention_heads": 24, "num_kv_heads": 4,
                               "expert_intermediate_size": 4224},
                            num_layers=10),
        description="Wide & Shallow (1536h x 10L, more width)",
    )

    # 7. Many Heads (~314M) - 32 heads, head_dim=32
    variants["v07_many_heads"] = VariantInfo(
        config=HamnerConfig(**{**_BASE, "num_attention_heads": 32, "num_kv_heads": 8},
                            num_layers=24),
        description="Many Heads (32 heads, 8 KV, finer attention)",
    )

    # 8. Emotional Transformer (~321M) - slow middle layers
    variants["v08_emotional"] = VariantInfo(
        config=HamnerConfig(**_BASE, num_layers=24,
                            emotional_layers=7, emotional_lr_scale=0.2),
        description="Emotional Transformer (7 slow middle layers)",
        param_group_fn=emotional_param_groups,
    )

    # 9. Multi-Query Attention (~305M) - kv_heads=1, fast inference
    variants["v09_mqa"] = VariantInfo(
        config=HamnerConfig(**{**_BASE, "num_kv_heads": 1},
                            num_layers=26),
        description="Multi-Query Attention (kv_heads=1, lean KV)",
    )

    # 10. Big MoE 16-expert (~335M total) - even more sparse capacity
    variants["v10_big_moe"] = VariantInfo(
        config=HamnerConfig(**{**_BASE, "num_experts": 16, "num_active_experts": 2,
                               "expert_intermediate_size": 640},
                            num_layers=8),
        description="Big MoE 16-expert top-2 (max sparse capacity)",
    )

    return variants


def describe_variant(name, variant_info):
    """Return a human-readable description string."""
    cfg = variant_info.config
    return f"{name}: {variant_info.description} (L={cfg.num_layers}, h={cfg.hidden_size})"

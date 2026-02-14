"""
10 Architecture Variants for Tournament Training
=================================================
Each variant explores a different idea. They compete on val loss.

1.  dense_small       - 120M dense, standard attention
2.  dense_medium      - 250M dense, standard attention
3.  diffattn_small    - 120M dense, differential attention
4.  diffattn_medium   - 250M dense, differential attention
5.  moe_small         - 300M total, standard attn + MoE
6.  moe_medium        - 500M total, standard attn + MoE
7.  hybrid_small      - 300M total, diff attn + MoE
8.  hybrid_medium     - 500M total, diff attn + MoE
9.  deep_narrow       - 150M, 32 layers, narrow
10. wide_shallow      - 150M, 8 layers, wide
"""

from model import HamnerConfig


def get_variant_configs() -> dict:
    variants = {}

    # 1. Dense Small (~120M)
    variants["v01_dense_small"] = HamnerConfig(
        hidden_size=512, num_layers=16, num_attention_heads=8, num_kv_heads=4,
        num_experts=1, num_active_experts=1, expert_intermediate_size=1536,
        use_differential_attention=False, gradient_checkpointing=True,
    )

    # 2. Dense Medium (~250M)
    variants["v02_dense_medium"] = HamnerConfig(
        hidden_size=768, num_layers=20, num_attention_heads=12, num_kv_heads=4,
        num_experts=1, num_active_experts=1, expert_intermediate_size=2048,
        use_differential_attention=False, gradient_checkpointing=True,
    )

    # 3. DiffAttn Small (~150M - larger Q/K projections)
    variants["v03_diffattn_small"] = HamnerConfig(
        hidden_size=512, num_layers=16, num_attention_heads=8, num_kv_heads=4,
        num_experts=1, num_active_experts=1, expert_intermediate_size=1536,
        use_differential_attention=True, gradient_checkpointing=True,
    )

    # 4. DiffAttn Medium (~300M)
    variants["v04_diffattn_medium"] = HamnerConfig(
        hidden_size=768, num_layers=20, num_attention_heads=12, num_kv_heads=4,
        num_experts=1, num_active_experts=1, expert_intermediate_size=2048,
        use_differential_attention=True, gradient_checkpointing=True,
    )

    # 5. MoE Small (~300M total, ~100M active)
    variants["v05_moe_small"] = HamnerConfig(
        hidden_size=512, num_layers=12, num_attention_heads=8, num_kv_heads=4,
        num_experts=8, num_active_experts=2, expert_intermediate_size=1024,
        use_differential_attention=False, gradient_checkpointing=True,
    )

    # 6. MoE Medium (~550M total, ~180M active)
    variants["v06_moe_medium"] = HamnerConfig(
        hidden_size=768, num_layers=16, num_attention_heads=12, num_kv_heads=4,
        num_experts=8, num_active_experts=2, expert_intermediate_size=1536,
        use_differential_attention=False, gradient_checkpointing=True,
    )

    # 7. Hybrid Small - DiffAttn + MoE (~350M total)
    variants["v07_hybrid_small"] = HamnerConfig(
        hidden_size=512, num_layers=12, num_attention_heads=8, num_kv_heads=4,
        num_experts=8, num_active_experts=2, expert_intermediate_size=1024,
        use_differential_attention=True, gradient_checkpointing=True,
    )

    # 8. Hybrid Medium - DiffAttn + MoE (~600M total)
    variants["v08_hybrid_medium"] = HamnerConfig(
        hidden_size=768, num_layers=16, num_attention_heads=12, num_kv_heads=4,
        num_experts=8, num_active_experts=2, expert_intermediate_size=1536,
        use_differential_attention=True, gradient_checkpointing=True,
    )

    # 9. Deep Narrow (~150M, lots of layers)
    variants["v09_deep_narrow"] = HamnerConfig(
        hidden_size=384, num_layers=32, num_attention_heads=6, num_kv_heads=3,
        num_experts=1, num_active_experts=1, expert_intermediate_size=1024,
        use_differential_attention=False, gradient_checkpointing=True,
    )

    # 10. Wide Shallow (~150M, few layers)
    variants["v10_wide_shallow"] = HamnerConfig(
        hidden_size=1024, num_layers=8, num_attention_heads=16, num_kv_heads=4,
        num_experts=1, num_active_experts=1, expert_intermediate_size=2048,
        use_differential_attention=False, gradient_checkpointing=True,
    )

    return variants


def describe_variant(name, config):
    is_diff = config.use_differential_attention
    is_moe = config.is_moe
    attn_type = "DiffAttn" if is_diff else "StdAttn"
    mlp_type = f"MoE-{config.num_experts}x top{config.num_active_experts}" if is_moe else "Dense"
    return (f"{name}: h={config.hidden_size} L={config.num_layers} "
            f"heads={config.num_attention_heads}/{config.num_kv_heads} "
            f"{attn_type} {mlp_type}")

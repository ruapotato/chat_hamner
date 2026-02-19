"""
Hamner Model Architecture
=========================
A novel hybrid architecture with pluggable components:
- Standard Attention OR Differential Attention
- Dense MLP OR Mixture of Experts MLP
- RoPE, RMSNorm, SwiGLU throughout

Supports 10 variant configurations for tournament training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HamnerConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 20
    num_attention_heads: int = 12
    num_kv_heads: int = 4
    num_experts: int = 8
    num_active_experts: int = 2
    expert_intermediate_size: int = 1536
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    router_aux_loss_coef: float = 0.01
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    use_differential_attention: bool = True  # toggle attention type

    # Architecture selector: transformer, mamba, rwkv, xlstm, hawk, griffin,
    #                        retnet, liquid, emotional, mamba_attn
    architecture: str = "transformer"

    # SSM / recurrent architecture settings
    d_state: int = 16           # state dimension for SSM models
    d_conv: int = 4             # conv width for Mamba
    expand_factor: int = 2      # expansion ratio for Mamba inner dim

    # Griffin / hybrid settings
    attn_every_n: int = 6       # insert attention layer every N blocks

    # Emotional transformer settings
    emotional_layers: int = 0   # number of middle layers with slower LR
    emotional_lr_scale: float = 0.2  # LR multiplier for emotional layers

    @property
    def is_moe(self):
        return self.num_experts > 1

    def total_params_estimate(self):
        embed = self.vocab_size * self.hidden_size
        head_dim = self.hidden_size // self.num_attention_heads
        kv_dim = self.num_kv_heads * head_dim
        if self.use_differential_attention:
            attn = self.hidden_size * self.hidden_size * 2 + self.hidden_size * kv_dim * 2 + self.hidden_size * kv_dim + self.hidden_size * self.hidden_size
        else:
            attn = self.hidden_size * self.hidden_size + self.hidden_size * kv_dim * 2 + self.hidden_size * self.hidden_size
        mlp = self.num_experts * self.hidden_size * self.expert_intermediate_size * 3
        if self.is_moe:
            mlp += self.hidden_size * self.num_experts
        per_layer = attn + mlp + self.hidden_size * 4
        total = embed + per_layer * self.num_layers + self.hidden_size
        if not self.tie_word_embeddings:
            total += embed
        return total


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos.repeat(1,1,1,2) + rotated * sin.repeat(1,1,1,2)


class StandardAttention(nn.Module):
    """Standard GQA attention with RoPE."""

    def __init__(self, config: HamnerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, rope_cos, rope_sin, attention_mask=None):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2,-1)) * scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out)


class DifferentialAttention(nn.Module):
    """
    Differential Attention (Microsoft DIFF Transformer, 2024).
    Computes two attention patterns and subtracts them to cancel noise.
    """

    def __init__(self, config: HamnerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # 2x projections for Q and K (differential pairs)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim * 2, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Learnable lambda
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx + 1))

        self.subln = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, rope_cos, rope_sin, attention_mask=None):
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads * 2, self.head_dim)
        q1, q2 = q[:,:,:self.num_heads,:].transpose(1,2), q[:,:,self.num_heads:,:].transpose(1,2)

        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads * 2, self.head_dim)
        k1, k2 = k[:,:,:self.num_kv_heads,:].transpose(1,2), k[:,:,self.num_kv_heads:,:].transpose(1,2)

        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)

        q1, q2 = apply_rope(q1, rope_cos, rope_sin), apply_rope(q2, rope_cos, rope_sin)
        k1, k2 = apply_rope(k1, rope_cos, rope_sin), apply_rope(k2, rope_cos, rope_sin)

        if self.num_kv_groups > 1:
            k1 = k1.repeat_interleave(self.num_kv_groups, dim=1)
            k2 = k2.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn1 = torch.matmul(q1, k1.transpose(-2,-1)) * scale
        attn2 = torch.matmul(q2, k2.transpose(-2,-1)) * scale

        if attention_mask is not None:
            attn1 = attn1 + attention_mask
            attn2 = attn2 + attention_mask

        attn1 = F.softmax(attn1, dim=-1, dtype=torch.float32).type_as(q1)
        attn2 = F.softmax(attn2, dim=-1, dtype=torch.float32).type_as(q2)

        lam = (torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
               - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
               + self.lambda_init)

        attn = attn1 - lam * attn2
        out = self.subln(torch.matmul(attn, v))
        out = out.transpose(1,2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out)


class ExpertMLP(nn.Module):
    """Single SwiGLU expert."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DenseMLP(nn.Module):
    """Single dense SwiGLU MLP (no routing)."""
    def __init__(self, config: HamnerConfig):
        super().__init__()
        self.mlp = ExpertMLP(config.hidden_size, config.expert_intermediate_size)

    def forward(self, x):
        return self.mlp(x), torch.tensor(0.0, device=x.device, dtype=x.dtype)


class MoELayer(nn.Module):
    """Mixture of Experts with top-k routing and load balancing loss."""

    def __init__(self, config: HamnerConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_active_experts
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertMLP(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.aux_loss_coef = config.router_aux_loss_coef

    def forward(self, x):
        bsz, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        num_tokens = x_flat.shape[0]

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            expert_out = expert(x_flat[idx])
            ep = torch.zeros(idx.shape[0], device=x.device, dtype=topk_probs.dtype)
            for k in range(self.top_k):
                km = topk_indices[idx, k] == i
                ep[km] = topk_probs[idx[km], k]
            output[idx] += expert_out * ep.unsqueeze(-1).type_as(expert_out)

        # Load balancing loss
        tpe = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)
        for k in range(self.top_k):
            tpe.scatter_add_(0, topk_indices[:,k], torch.ones(num_tokens, device=x.device, dtype=torch.float32))
        tpe = tpe / (num_tokens * self.top_k)
        rppe = router_probs.mean(dim=0)
        aux_loss = self.num_experts * (tpe * rppe).sum() * self.aux_loss_coef

        return output.view(bsz, seq_len, hidden), aux_loss


class HamnerBlock(nn.Module):
    def __init__(self, config: HamnerConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_differential_attention:
            self.attention = DifferentialAttention(config, layer_idx)
        else:
            self.attention = StandardAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.is_moe:
            self.mlp = MoELayer(config)
        else:
            self.mlp = DenseMLP(config)

    def forward(self, x, rope_cos, rope_sin, attention_mask=None):
        residual = x
        x = self.attention(self.attn_norm(x), rope_cos, rope_sin, attention_mask)
        x = residual + x
        residual = x
        mlp_out, aux_loss = self.mlp(self.mlp_norm(x))
        x = residual + mlp_out
        return x, aux_loss


class HamnerModel(nn.Module):
    def __init__(self, config: HamnerConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        head_dim = config.hidden_size // config.num_attention_heads
        rope_cos, rope_sin = precompute_rope(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.layers = nn.ModuleList([HamnerBlock(config, i) for i in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)
        self._gradient_checkpointing = config.gradient_checkpointing

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        x = self.embed_tokens(input_ids)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # Build pad mask: 0 for valid tokens, -inf for padding
            # Note: can't use (1-mask)*-inf because 0*-inf = NaN in IEEE 754
            pad_mask = torch.zeros(bsz, 1, 1, seq_len, device=device, dtype=x.dtype)
            pad_mask.masked_fill_(attention_mask[:, None, None, :] == 0, float("-inf"))
            causal_mask = causal_mask + pad_mask

        total_aux_loss = torch.tensor(0.0, device=device, dtype=x.dtype)
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x, aux_loss = torch.utils.checkpoint.checkpoint(
                    layer, x, self.rope_cos, self.rope_sin, causal_mask, use_reentrant=False,
                )
            else:
                x, aux_loss = layer(x, self.rope_cos, self.rope_sin, causal_mask)
            total_aux_loss = total_aux_loss + aux_loss

        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss = loss + total_aux_loss

        return {"loss": loss, "logits": logits, "aux_loss": total_aux_loss}

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1, eos_token_id=2):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits = self(idx_cond)["logits"][:, -1, :]
            if repetition_penalty != 1.0:
                for tid in set(input_ids[0].tolist()):
                    logits[0, tid] /= repetition_penalty
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                rm = cp > top_p
                rm[..., 1:] = rm[..., :-1].clone()
                rm[..., 0] = False
                logits[rm.scatter(1, si, rm)] = float("-inf")
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break
        return input_ids


# ---------------------------------------------------------------------------
# Base class for non-transformer language models
# ---------------------------------------------------------------------------

class BaseLanguageModel(nn.Module):
    """Shared skeleton: embedding -> blocks -> norm -> lm_head.

    Subclasses set self.blocks in __init__ then call self._post_init().
    """

    def __init__(self, config: HamnerConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.blocks = nn.ModuleList()  # set by subclass before _post_init()
        self._gradient_checkpointing = config.gradient_checkpointing

    def _post_init(self):
        """Call after subclass has set self.blocks."""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None):
        x = self.embed_tokens(input_ids)
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits,
                "aux_loss": torch.tensor(0.0, device=input_ids.device)}

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8,
                 top_k=50, top_p=0.9, repetition_penalty=1.1, eos_token_id=2):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits = self(idx_cond)["logits"][:, -1, :]
            if repetition_penalty != 1.0:
                for tid in set(input_ids[0].tolist()):
                    logits[0, tid] /= repetition_penalty
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                rm = cp > top_p
                rm[..., 1:] = rm[..., :-1].clone()
                rm[..., 0] = False
                logits[rm.scatter(1, si, rm)] = float("-inf")
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break
        return input_ids


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(config: HamnerConfig):
    """Create the appropriate model for the given architecture config."""
    arch = config.architecture
    if arch in ("transformer", "emotional"):
        return HamnerModel(config)

    # Import alternative architectures (keeps model.py clean)
    from architectures import (
        MambaModel, RWKVModel, XLSTMModel, HawkModel,
        GriffinModel, RetNetModel, LiquidModel, MambaAttentionModel,
    )
    registry = {
        "mamba": MambaModel,
        "rwkv": RWKVModel,
        "xlstm": XLSTMModel,
        "hawk": HawkModel,
        "griffin": GriffinModel,
        "retnet": RetNetModel,
        "liquid": LiquidModel,
        "mamba_attn": MambaAttentionModel,
    }
    if arch not in registry:
        raise ValueError(f"Unknown architecture: {arch}. "
                         f"Choose from: {list(registry.keys())}")
    return registry[arch](config)

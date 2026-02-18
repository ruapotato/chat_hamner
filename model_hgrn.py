"""
HGRN2 Hybrid Language Model
============================
A hybrid architecture combining:
  - HGRN2 (Hierarchically Gated Recurrent Neural Network v2) layers for most blocks
  - Sparse full GQA attention with RoPE every Nth layer

Based on:
  - "HGRN2: Gated Linear RNNs with State Expansion" (Qin et al., 2024)
  - Hybrid linear/full attention pattern from GatedDeltaNet / Qwen3-Next

Core HGRN2 recurrence (Gated Linear Attention interpretation):
  S_t = diag(g_t) * S_{t-1} + v_t * k_t^T     (gated outer-product state update)
  o_t = S_t @ q_t                               (query retrieval from state)

where:
  g_t = sigmoid(f_t) is the forget gate (data-dependent decay)
  k_t = 1 - g_t (key derived from complement of forget gate, saves parameters)
  q_t = swish(q_proj(x)) is the query (output gate in RNN terminology)
  v_t = i_proj(x) is the value (input in RNN terminology)

Hybrid strategy: every 6th layer (indices 5, 11, 17, 23) uses standard grouped-query
attention with RoPE, giving the model periodic access to precise token-to-token
attention while linear layers handle bulk sequence processing efficiently.

Pure PyTorch -- no custom CUDA/Triton kernels required.
Target: ~340M parameters with vocab_size=49152.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from dataclasses import dataclass
from typing import Optional, List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HGRN2Config:
    vocab_size: int = 49152
    hidden_size: int = 1024
    num_layers: int = 24
    max_seq_len: int = 1024
    gradient_checkpointing: bool = True

    # Full attention settings (for sparse attention layers)
    num_attention_heads: int = 16     # Q heads
    num_kv_heads: int = 4             # KV heads (GQA)
    attn_every_n: int = 6            # full attention every Nth layer

    # HGRN2-specific
    expand_ratio: int = 128           # state expansion per head (head_dim for recurrent state)
    # num_hgrn_heads is derived: hidden_size // expand_ratio

    # MLP
    intermediate_size: int = 2688     # SwiGLU intermediate; yields ~343M total params
    mlp_bias: bool = False

    # Normalization
    rms_norm_eps: float = 1e-5

    # RoPE
    rope_theta: float = 10000.0

    # Weight tying
    tie_word_embeddings: bool = True

    # Initialization
    initializer_range: float = 0.02

    @property
    def num_hgrn_heads(self) -> int:
        """Number of heads for the HGRN2 recurrent layers."""
        return self.hidden_size // self.expand_ratio

    @property
    def head_dim(self) -> int:
        """Head dimension for full attention layers."""
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        """Total KV dimension for GQA attention layers."""
        return self.num_kv_heads * self.head_dim

    def full_attention_layer_indices(self) -> List[int]:
        """Return 0-indexed layer indices that use full attention."""
        return [i for i in range(self.num_layers) if (i + 1) % self.attn_every_n == 0]

    def total_params_estimate(self) -> int:
        """Estimate total parameter count without instantiating the model."""
        embed = self.vocab_size * self.hidden_size

        # HGRN2 layer: q_proj + f_proj + i_proj + o_proj (all hidden->hidden) + g_norm
        hgrn2_attn = self.hidden_size * self.hidden_size * 4 + self.hidden_size

        # Full attention layer: q_proj(h->h) + k_proj(h->kv) + v_proj(h->kv) + o_proj(h->h)
        full_attn = (self.hidden_size * self.hidden_size
                     + self.hidden_size * self.kv_dim * 2
                     + self.hidden_size * self.hidden_size)

        # SwiGLU MLP: gate_proj + up_proj + down_proj (each h*inter)
        mlp = self.hidden_size * self.intermediate_size * 3

        # 2 RMSNorm per layer
        norms_per_layer = 2 * self.hidden_size

        full_attn_indices = self.full_attention_layer_indices()
        num_full = len(full_attn_indices)
        num_hgrn = self.num_layers - num_full

        total = embed  # embedding (tied with lm_head)
        total += num_hgrn * (hgrn2_attn + mlp + norms_per_layer)
        total += num_full * (full_attn + mlp + norms_per_layer)
        total += self.hidden_size  # final RMSNorm

        if not self.tie_word_embeddings:
            total += embed  # separate lm_head

        return total


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        norm = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * norm).to(dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embedding (for full attention layers)
# ---------------------------------------------------------------------------

def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute cos and sin for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x of shape (B, num_heads, seq_len, head_dim)."""
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos.repeat(1, 1, 1, 2) + rotated * sin.repeat(1, 1, 1, 2)


# ---------------------------------------------------------------------------
# HGRN2 Attention Layer (Gated Linear Attention with State Expansion)
# ---------------------------------------------------------------------------

class HGRN2Attention(nn.Module):
    """
    HGRN2 recurrent layer with gated linear attention.

    The recurrence:
        g_t = sigmoid(f_proj(x_t))               -- forget gate
        k_t = 1 - g_t                            -- key (complement of forget)
        q_t = swish(q_proj(x_t))                 -- query / output gate
        v_t = i_proj(x_t)                        -- value / input

    State update (per head, with state expansion via outer product):
        S_t = diag(g_t) * S_{t-1} + v_t @ k_t^T   -- (expand_ratio x head_i_dim) matrix
        o_t = q_t^T @ S_t                          -- (head_i_dim,) vector

    This is equivalent to gated linear attention (GLA) where the forget gate
    provides data-dependent exponential decay on the recurrent state.

    For training efficiency, we use a parallel (non-recurrent) formulation
    that computes the full sequence output via cumulative gated attention.
    """

    def __init__(self, config: HGRN2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_hgrn_heads       # e.g. 8
        self.expand_ratio = config.expand_ratio       # e.g. 128 (acts as head_dim for q/f/k)
        self.layer_idx = layer_idx

        # Dimensions per head
        self.head_f_dim = self.expand_ratio                    # 128 (forget/query/key dim)
        self.head_i_dim = config.hidden_size // self.num_heads # 128 (value dim)

        # Total projection dimensions
        self.forget_dim = self.num_heads * self.head_f_dim  # = hidden_size typically

        # Projections
        # q_proj: query (output gate), activated with swish
        self.q_proj = nn.Linear(config.hidden_size, self.forget_dim, bias=False)
        # f_proj: forget gate, activated with sigmoid (via logsigmoid for stability)
        self.f_proj = nn.Linear(config.hidden_size, self.forget_dim, bias=False)
        # i_proj: input (value stream)
        self.i_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # o_proj: output projection
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Group norm on output (before output projection)
        self.g_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def _gla_forward(self, q, f, v):
        """GLA computation in float32 to avoid FP16 overflow in cumulative gates."""
        B, T, D = q.shape

        log_g = F.logsigmoid(f)

        q = q.view(B, T, self.num_heads, self.head_f_dim)
        log_g = log_g.view(B, T, self.num_heads, self.head_f_dim)
        v = v.view(B, T, self.num_heads, self.head_i_dim)

        k = 1.0 - log_g.exp()

        o = self._parallel_gated_linear_attention(q, k, v, log_g)
        return o.reshape(B, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        B, T, D = x.shape

        # Compute projections
        q = F.silu(self.q_proj(x))
        f = self.f_proj(x)
        v = self.i_proj(x)

        # GLA computation in float32 (cumulative gates overflow FP16)
        o = self._gla_forward(q, f, v)

        # Normalize and project
        o = self.g_norm(o)
        o = self.o_proj(o)
        return o

    def _parallel_gated_linear_attention(
        self,
        q: torch.Tensor,    # (B, T, H, Df)
        k: torch.Tensor,    # (B, T, H, Df)
        v: torch.Tensor,    # (B, T, H, Di)
        log_g: torch.Tensor  # (B, T, H, Df)
    ) -> torch.Tensor:
        """
        Chunk-based parallel computation of gated linear attention.

        The recurrence S_t = diag(g_t) * S_{t-1} + v_t * k_t^T can be split into:
          - Cross-chunk: contribution from state accumulated in prior chunks
          - Intra-chunk: contribution from tokens within the current chunk

        For intra-chunk, we absorb cumulative gates into q and k, then compute
        standard dot-product attention (CHUNK x CHUNK matrix per head).
        Within a 64-step chunk, the cumulative gates stay in float32 safe range.
        For cross-chunk, we use efficient einsum operations with the carried state.
        """
        B, T, H, Df = q.shape
        Di = v.shape[-1]

        # Choose chunk size -- balance between memory and parallelism
        CHUNK = 64
        if T <= CHUNK:
            return self._gla_recurrent_chunk(q, k, v, log_g)

        # Pad sequence to multiple of CHUNK
        pad = (CHUNK - T % CHUNK) % CHUNK
        if pad > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
            log_g = F.pad(log_g, (0, 0, 0, 0, 0, pad))

        T_padded = q.shape[1]
        num_chunks = T_padded // CHUNK

        # Reshape into chunks: (B, num_chunks, CHUNK, H, D)
        q_c = q.view(B, num_chunks, CHUNK, H, Df)
        k_c = k.view(B, num_chunks, CHUNK, H, Df)
        v_c = v.view(B, num_chunks, CHUNK, H, Di)
        log_g_c = log_g.view(B, num_chunks, CHUNK, H, Df)

        outputs = []

        # State carried across chunks: (B, H, Df, Di)
        state = q.new_zeros(B, H, Df, Di)

        for chunk_idx in range(num_chunks):
            q_ch = q_c[:, chunk_idx]      # (B, CHUNK, H, Df)
            k_ch = k_c[:, chunk_idx]      # (B, CHUNK, H, Df)
            v_ch = v_c[:, chunk_idx]      # (B, CHUNK, H, Di)
            log_g_ch = log_g_c[:, chunk_idx]  # (B, CHUNK, H, Df)

            # --- Cross-chunk: apply accumulated state ---
            # Cumulative gate from start of chunk to each position
            cum_log_g = log_g_ch.cumsum(dim=1)  # (B, CHUNK, H, Df)

            # Decay the carried state by the cumulative gate at each position
            q_decayed = q_ch * cum_log_g.exp()  # (B, CHUNK, H, Df)

            # (B, CHUNK, H, Df) @ (B, H, Df, Di) -> (B, CHUNK, H, Di)
            cross_out = torch.einsum('bthr,bhrd->bthd', q_decayed, state)

            # --- Intra-chunk: recurrent within the chunk ---
            intra_out = self._gla_recurrent_chunk(q_ch, k_ch, v_ch, log_g_ch)

            # Combine
            chunk_out = intra_out + cross_out
            outputs.append(chunk_out)

            # --- Update state for next chunk ---
            total_gate = cum_log_g[:, -1:, :, :]  # (B, 1, H, Df)

            # Decay old state
            state = state * total_gate.squeeze(1).exp().unsqueeze(-1)

            # Accumulate new contributions:
            # For position t, gate from t to end = exp(cum_log_g[-1] - cum_log_g[t])
            reverse_gate = (total_gate - cum_log_g).exp()  # (B, CHUNK, H, Df)
            k_gated = k_ch * reverse_gate

            # Outer product sum: (B, CHUNK, H, Df) x (B, CHUNK, H, Di) -> (B, H, Df, Di)
            state = state + torch.einsum('bthr,bthd->bhrd', k_gated, v_ch)

        # Concatenate all chunk outputs
        o = torch.cat(outputs, dim=1)  # (B, T_padded, H, Di)

        # Remove padding
        if pad > 0:
            o = o[:, :T, :, :]

        return o

    def _gla_recurrent_chunk(
        self,
        q: torch.Tensor,    # (B, T, H, Df)
        k: torch.Tensor,    # (B, T, H, Df)
        v: torch.Tensor,    # (B, T, H, Di)
        log_g: torch.Tensor  # (B, T, H, Df)
    ) -> torch.Tensor:
        """
        Parallel intra-chunk gated linear attention.

        Absorbs cumulative log-gates into q and k so that the gated attention
        reduces to a simple dot-product attention:
            q_hat[t] = q[t] * exp(cum_log_g[t])
            k_hat[s] = k[s] * exp(-cum_log_g[s])
            attn[t,s] = q_hat[t] . k_hat[s]   (dot product over Df)

        Within a chunk of ~64 steps, cumulative log-gates range from ~0 to ~-40,
        so exp values stay well within float32 safe range (~3.4e38).

        Fallback: for very short sequences (T <= 16, e.g. during generation),
        uses a simple recurrent loop which avoids any materialization.
        """
        B, T, H, Df = q.shape
        Di = v.shape[-1]

        if T <= 16:
            return self._gla_recurrent_small(q, k, v, log_g)

        # Cumulative log gates: (B, T, H, Df)
        cum_log_g = log_g.cumsum(dim=1)

        # Compute in float32 for numerical stability
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        cum_f = cum_log_g.float()

        # Reshape for batched ops: merge B and H
        BH = B * H
        q_r = q_f.permute(0, 2, 1, 3).reshape(BH, T, Df)   # (BH, T, Df)
        k_r = k_f.permute(0, 2, 1, 3).reshape(BH, T, Df)   # (BH, T, Df)
        v_r = v_f.permute(0, 2, 1, 3).reshape(BH, T, Di)   # (BH, T, Di)
        cum_r = cum_f.permute(0, 2, 1, 3).reshape(BH, T, Df)  # (BH, T, Df)

        # Absorb cumulative gates into q and k:
        #   q_hat[t,d] = q[t,d] * exp(cum[t,d])
        #   k_hat[s,d] = k[s,d] * exp(-cum[s,d])
        #   attn[t,s] = sum_d q_hat[t,d] * k_hat[s,d]  (dot product)
        #
        # Within a chunk of ~64 steps, cum goes from ~0 to ~-40.
        # exp(-(-40)) = exp(40) ~ 2e17, well within float32 range (~3.4e38).
        q_hat = q_r * cum_r.exp()         # (BH, T, Df)
        k_hat = k_r * (-cum_r).exp()      # (BH, T, Df)

        # Attention weights: (BH, T, T) = q_hat @ k_hat^T
        attn = torch.bmm(q_hat, k_hat.transpose(-2, -1))

        # Causal mask: zero out future positions
        causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.float32))
        attn = attn * causal.unsqueeze(0)

        # Output: (BH, T, Di) = attn @ v
        o = torch.bmm(attn, v_r)

        # Reshape back: (B, H, T, Di) -> (B, T, H, Di)
        o = o.reshape(B, H, T, Di).permute(0, 2, 1, 3)
        return o.to(q.dtype)

    def _gla_recurrent_small(
        self,
        q: torch.Tensor,    # (B, T, H, Df)
        k: torch.Tensor,    # (B, T, H, Df)
        v: torch.Tensor,    # (B, T, H, Di)
        log_g: torch.Tensor  # (B, T, H, Df)
    ) -> torch.Tensor:
        """
        Simple recurrent GLA for very short sequences (T <= 16).
        Used during generation where each step processes 1 token.
        """
        B, T, H, Df = q.shape
        Di = v.shape[-1]

        g = log_g.exp()
        q_r = q.permute(0, 2, 1, 3)
        k_r = k.permute(0, 2, 1, 3)
        v_r = v.permute(0, 2, 1, 3)
        g_r = g.permute(0, 2, 1, 3)

        state = q.new_zeros(B, H, Df, Di)
        outputs = []

        for t in range(T):
            gate = g_r[:, :, t, :].unsqueeze(-1)
            state = state * gate
            state = state + k_r[:, :, t, :].unsqueeze(-1) * v_r[:, :, t, :].unsqueeze(-2)
            o_step = torch.einsum('bhrd,bhr->bhd', state, q_r[:, :, t, :])
            outputs.append(o_step)

        o = torch.stack(outputs, dim=2).permute(0, 2, 1, 3)
        return o


# ---------------------------------------------------------------------------
# Full Attention Layer (GQA with RoPE)
# ---------------------------------------------------------------------------

class FullAttention(nn.Module):
    """Standard grouped-query attention with RoPE for the sparse attention layers."""

    def __init__(self, config: HGRN2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # GQA: repeat KV heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: HGRN2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# HGRN2 Block (HGRN2 Attention + MLP)
# ---------------------------------------------------------------------------

class HGRN2Block(nn.Module):
    """Pre-norm block with HGRN2 gated linear attention."""

    def __init__(self, config: HGRN2Config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = HGRN2Attention(config, layer_idx)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # HGRN2 attention (does not use RoPE or causal mask -- recurrence is causal by nature)
        residual = x
        x = self.attention(self.attn_norm(x))
        x = residual + x

        # MLP
        residual = x
        x = self.mlp(self.mlp_norm(x))
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# Full Attention Block (GQA + MLP)
# ---------------------------------------------------------------------------

class FullAttentionBlock(nn.Module):
    """Pre-norm block with full GQA attention and RoPE."""

    def __init__(self, config: HGRN2Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = FullAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Full attention with RoPE and causal mask
        residual = x
        x = self.attention(self.attn_norm(x), rope_cos, rope_sin, attention_mask)
        x = residual + x

        # MLP
        residual = x
        x = self.mlp(self.mlp_norm(x))
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# HGRN2 Hybrid Language Model
# ---------------------------------------------------------------------------

class HGRN2Model(nn.Module):
    """
    HGRN2 hybrid language model.

    Most layers use HGRN2 gated linear attention (O(n) per layer).
    Every attn_every_n-th layer uses full GQA attention with RoPE (O(n^2) per layer).

    With default config (24 layers, attn_every_n=6):
      - Layers 0-4, 6-10, 12-16, 18-22: HGRN2 (20 layers)
      - Layers 5, 11, 17, 23: Full attention (4 layers)
    """

    def __init__(self, config: HGRN2Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # RoPE buffers (only used by full attention layers, but precomputed once)
        head_dim = config.head_dim
        rope_cos, rope_sin = precompute_rope(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # Build layers: mix of HGRN2 and full attention
        full_attn_indices = set(config.full_attention_layer_indices())
        layers = []
        for i in range(config.num_layers):
            if i in full_attn_indices:
                layers.append(FullAttentionBlock(config))
            else:
                layers.append(HGRN2Block(config, layer_idx=i))
        self.layers = nn.ModuleList(layers)

        # Final norm and head
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)
        self._gradient_checkpointing = config.gradient_checkpointing

        # Track which layers are full attention for the forward pass
        self._full_attn_indices = full_attn_indices

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) target token IDs for loss computation
            attention_mask: (batch, seq_len) padding mask (1 = attend, 0 = pad)

        Returns:
            dict with keys: "loss", "logits", "aux_loss"
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed_tokens(input_ids)

        # Causal mask for full attention layers
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device, dtype=x.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # Combine padding mask with causal mask
            # Use torch.where to avoid 0 * -inf = NaN
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            causal_mask = torch.where(
                pad_mask.bool(),
                causal_mask,
                torch.tensor(float("-inf"), device=device, dtype=x.dtype),
            )

        # Forward through all layers
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, self.rope_cos, self.rope_sin, causal_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(x, self.rope_cos, self.rope_sin, causal_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": torch.tensor(0.0, device=device, dtype=logits.dtype),
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k, top-p sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits = self(idx_cond)["logits"][:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                logits[remove_mask.scatter(1, sorted_indices, remove_mask)] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return input_ids


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("HGRN2 Hybrid Language Model -- Standalone Test")
    print("=" * 70)

    config = HGRN2Config()
    print(f"\nConfig:")
    print(f"  vocab_size       = {config.vocab_size}")
    print(f"  hidden_size      = {config.hidden_size}")
    print(f"  num_layers       = {config.num_layers}")
    print(f"  num_hgrn_heads   = {config.num_hgrn_heads}")
    print(f"  expand_ratio     = {config.expand_ratio}")
    print(f"  head_dim (attn)  = {config.head_dim}")
    print(f"  num_attn_heads   = {config.num_attention_heads}")
    print(f"  num_kv_heads     = {config.num_kv_heads}")
    print(f"  intermediate     = {config.intermediate_size}")
    print(f"  attn_every_n     = {config.attn_every_n}")
    print(f"  full attn layers = {config.full_attention_layer_indices()}")
    print(f"  max_seq_len      = {config.max_seq_len}")
    print(f"  estimated params = {config.total_params_estimate() / 1e6:.1f}M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Build model
    model = HGRN2Model(config).to(device)
    total, trainable = model.count_parameters()
    print(f"Actual total params:     {total / 1e6:.1f}M ({total:,})")
    print(f"Actual trainable params: {trainable / 1e6:.1f}M ({trainable:,})")

    # Test forward pass
    print("\n--- Forward pass test ---")
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    output = model(input_ids, labels=labels)
    print(f"  logits shape: {output['logits'].shape}")
    print(f"  loss:         {output['loss'].item():.4f}")
    print(f"  aux_loss:     {output['aux_loss'].item():.4f}")

    # Test with attention mask
    print("\n--- Forward pass with attention mask ---")
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    attention_mask[:, -10:] = 0  # mask last 10 tokens
    output_masked = model(input_ids, labels=labels, attention_mask=attention_mask)
    print(f"  logits shape: {output_masked['logits'].shape}")
    print(f"  loss:         {output_masked['loss'].item():.4f}")

    # Test generation
    print("\n--- Generation test ---")
    prompt = torch.randint(0, config.vocab_size, (1, 16), device=device)
    generated = model.generate(prompt, max_new_tokens=32, temperature=1.0, top_k=50)
    print(f"  prompt length:    {prompt.shape[1]}")
    print(f"  generated length: {generated.shape[1]}")
    print(f"  new tokens:       {generated.shape[1] - prompt.shape[1]}")

    # Test gradient checkpointing
    print("\n--- Gradient checkpointing test ---")
    config_gc = HGRN2Config(gradient_checkpointing=True)
    model_gc = HGRN2Model(config_gc).to(device)
    model_gc.train()
    output_gc = model_gc(input_ids, labels=labels)
    output_gc["loss"].backward()
    print(f"  loss with grad ckpt: {output_gc['loss'].item():.4f}")
    print(f"  grad computed successfully: True")

    # Layer breakdown
    print("\n--- Layer breakdown ---")
    for i, layer in enumerate(model.layers):
        layer_type = "FullAttention" if i in model._full_attn_indices else "HGRN2"
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"  Layer {i:2d}: {layer_type:14s}  params={layer_params / 1e6:.2f}M")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

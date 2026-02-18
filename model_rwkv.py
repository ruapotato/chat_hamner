"""
RWKV-7 "Goose" Language Model (x070) -- Pure PyTorch Implementation
====================================================================
Based on BlinkDL/RWKV-LM reference code (https://github.com/BlinkDL/RWKV-LM)
and the RWKV-7 paper: "RWKV-7 Goose with Expressive Dynamic State Evolution"
(https://arxiv.org/abs/2503.14456)

Key RWKV-7 innovations implemented here:
  - Data-dependent state transition via generalized delta rule
  - Separated removal/replacement keys (-kk, kk*a)
  - In-context learning rate 'a' (sigmoid-gated, LoRA-parameterized)
  - Value residual across layers (v_first from layer 0 blended into later layers)
  - Token shift (lerp between x[t-1] and x[t]) for temporal mixing
  - Exponential decay weights w with soft-clamped LoRA parameterization
  - Channel mixing with squared-ReLU activation
  - GroupNorm after WKV, plus bonus r*k*v term

This file uses the PARALLEL (training-mode) formulation of the WKV
recurrence so it can be trained efficiently without custom CUDA kernels.
It also provides a sequential generate() method for autoregressive inference.

Target: ~350M parameters with vocab_size=49152, hidden_size=1024, 24 layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RWKVConfig:
    vocab_size: int = 49152
    hidden_size: int = 960             # 960x22 = ~350M params
    num_layers: int = 22
    max_seq_len: int = 1024
    gradient_checkpointing: bool = True

    # RWKV-specific
    head_size: int = 64                # 960/64 = 15 heads
    # num_heads is derived: hidden_size // head_size

    # LoRA ranks for the various data-dependent projections
    # Following the RWKV-7 convention: max(32, round(factor * sqrt(C) / 32) * 32)
    decay_lora_rank: int = 64          # D_DECAY_LORA
    aaa_lora_rank: int = 64            # D_AAA_LORA  (in-context learning rate)
    mv_lora_rank: int = 32             # D_MV_LORA   (value residual mixing)
    gate_lora_rank: int = 128          # D_GATE_LORA

    # FFN expansion ratio
    ffn_expansion: int = 4

    @property
    def num_heads(self) -> int:
        return self.hidden_size // self.head_size

    @property
    def dim_att(self) -> int:
        return self.hidden_size

    @property
    def dim_ffn(self) -> int:
        return self.hidden_size * self.ffn_expansion

    def total_params_estimate(self) -> int:
        C = self.hidden_size
        V = self.vocab_size
        L = self.num_layers
        H = self.num_heads
        N = self.head_size
        ffn = self.dim_ffn

        # Embedding + LM head (untied)
        embed = V * C * 2  # emb + head

        # Per-layer time-mixing params
        # 6 token-shift lerp vectors: x_r, x_w, x_k, x_v, x_a, x_g  -> 6*C
        # w0(C) + w1(C, decay_lora) + w2(decay_lora, C)
        # a0(C) + a1(C, aaa_lora) + a2(aaa_lora, C)
        # v0(C) + v1(C, mv_lora) + v2(mv_lora, C)
        # g1(C, gate_lora) + g2(gate_lora, C)
        # k_k(C) + k_a(C) + r_k(H, N)
        # receptance(C, C) + key(C, C) + value(C, C) + output(C, C) = 4*C*C
        # GroupNorm: 2*C (weight + bias)
        tmix_lerp = 6 * C
        tmix_w = C + C * self.decay_lora_rank + self.decay_lora_rank * C
        tmix_a = C + C * self.aaa_lora_rank + self.aaa_lora_rank * C
        tmix_v = C + C * self.mv_lora_rank + self.mv_lora_rank * C
        tmix_g = C * self.gate_lora_rank + self.gate_lora_rank * C
        tmix_misc = C + C + H * N  # k_k, k_a, r_k
        tmix_linear = 4 * C * C
        tmix_norm = 2 * C
        tmix = tmix_lerp + tmix_w + tmix_a + tmix_v + tmix_g + tmix_misc + tmix_linear + tmix_norm

        # Per-layer channel-mixing params
        # x_k(C) + key(C, ffn) + value(ffn, C)
        cmix = C + C * ffn + ffn * C

        # Per-layer norms: ln0(first layer only, 2*C) + ln1(2*C) + ln2(2*C)
        norms = 2 * C * 2  # ln1 + ln2 per layer
        norms_first = 2 * C  # ln0 only on first layer

        per_layer = tmix + cmix + norms
        total = embed + per_layer * L + norms_first + 2 * C  # final ln_out
        return total


# ---------------------------------------------------------------------------
# Pure-PyTorch parallel WKV-7 computation (no CUDA kernels)
# ---------------------------------------------------------------------------

def rwkv7_attn_parallel(r, w, k, v, a, b, head_size):
    """
    Parallel (training-mode) RWKV-7 WKV computation in pure PyTorch.

    The recurrence is:
        state[t] = state[t-1] * diag(w[t]) + state[t-1] @ (a[t] @ b[t]^T) + v[t] @ k[t]^T
        out[t]   = state[t] @ r[t]

    where a = -kk (removal key), b = kk * alpha (replacement key).

    For training we unroll this sequentially over time -- this is O(T * H * N^2)
    which is the same complexity as the recurrence, just without custom CUDA.

    Args:
        r: (B, T, C) receptance
        w: (B, T, C) log-space decay (will be exponentiated)
        k: (B, T, C) keys
        v: (B, T, C) values
        a: (B, T, C) removal key  (= -kk)
        b: (B, T, C) replacement key (= kk * alpha)
        head_size: int

    Returns:
        out: (B, T, C) output
    """
    B, T, C = r.shape
    H = C // head_size
    N = head_size

    # Reshape to (B, T, H, N)
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-F.softplus(-w.view(B, T, H, N).float()) - 0.5)  # maps from log-domain to (0, exp(-0.5))

    out = torch.zeros(B, T, H, N, device=r.device, dtype=torch.float32)
    state = torch.zeros(B, H, N, N, device=r.device, dtype=torch.float32)

    for t in range(T):
        rt = r[:, t]                              # (B, H, N)
        kt = k[:, t].unsqueeze(-2)                # (B, H, 1, N)
        vt = v[:, t].unsqueeze(-1)                # (B, H, N, 1)
        at = a[:, t].unsqueeze(-1)                # (B, H, N, 1)
        bt = b[:, t].unsqueeze(-2)                # (B, H, 1, N)
        wt = w[:, t].unsqueeze(-2)                # (B, H, 1, N) -- broadcast over rows

        # State update: S = S * diag(w) + S @ (a @ b^T) + v @ k^T
        state = state * wt + state @ (at @ bt) + vt @ kt

        # Output: o = S @ r
        out[:, t] = (state @ rt.unsqueeze(-1)).squeeze(-1)  # (B, H, N)

    return out.view(B, T, C)


def rwkv7_attn_sequential(r, w, k, v, a, b, state, head_size):
    """
    Single-step RNN-mode WKV for autoregressive generation.

    Args:
        r, w, k, v, a, b: all (B, 1, C) -- single time step
        state: (B, H, N, N) -- recurrent state
        head_size: int

    Returns:
        out: (B, 1, C)
        new_state: (B, H, N, N)
    """
    B, _, C = r.shape
    H = C // head_size
    N = head_size

    r = r.view(B, H, N).float()
    k = k.view(B, H, N).float()
    v = v.view(B, H, N).float()
    a = a.view(B, H, N).float()
    b = b.view(B, H, N).float()
    w = torch.exp(-F.softplus(-w.view(B, H, N).float()) - 0.5)

    kt = k.unsqueeze(-2)   # (B, H, 1, N)
    vt = v.unsqueeze(-1)   # (B, H, N, 1)
    at = a.unsqueeze(-1)   # (B, H, N, 1)
    bt = b.unsqueeze(-2)   # (B, H, 1, N)
    wt = w.unsqueeze(-2)   # (B, H, 1, N)

    state = state.float()
    state = state * wt + state @ (at @ bt) + vt @ kt

    out = (state @ r.unsqueeze(-1)).squeeze(-1)  # (B, H, N)
    return out.view(B, 1, C), state


# ---------------------------------------------------------------------------
# RWKV-7 Time Mixing (Attention replacement)
# ---------------------------------------------------------------------------

class RWKV_TimeMix(nn.Module):
    """
    RWKV-7 Time Mixing block (x070).

    Implements the full data-dependent state evolution with:
    - Token shift via learnable lerp parameters
    - LoRA-parameterized decay (w), in-context learning rate (a),
      value residual (v), and gate (g)
    - Normalized keys (kk) for the delta-rule update
    - GroupNorm after WKV + bonus r*k*v term
    """

    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.head_size = config.head_size
        self.n_head = config.num_heads
        C = config.hidden_size
        H = self.n_head
        N = self.head_size

        assert C % N == 0
        assert C == H * N

        # Token-shift lerp parameters (mu in the paper)
        # x_shifted = x + (x_prev - x) * mu = lerp(x, x_prev, mu)
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        # Decay (w): w0 + LoRA(xw)  -> soft-clamped via -softplus(-x) - 0.5
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, config.decay_lora_rank))
        self.w2 = nn.Parameter(torch.empty(config.decay_lora_rank, C))

        # In-context learning rate (a): sigmoid(a0 + LoRA(xa))
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, config.aaa_lora_rank))
        self.a2 = nn.Parameter(torch.empty(config.aaa_lora_rank, C))

        # Value residual mixing (v): sigmoid(v0 + LoRA(xv)) -- layers > 0 only
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, config.mv_lora_rank))
        self.v2 = nn.Parameter(torch.empty(config.mv_lora_rank, C))

        # Gate (g): sigmoid(xg @ g1) @ g2
        self.g1 = nn.Parameter(torch.empty(C, config.gate_lora_rank))
        self.g2 = nn.Parameter(torch.empty(config.gate_lora_rank, C))

        # Key normalization and scaling
        self.k_k = nn.Parameter(torch.empty(1, 1, C))  # element-wise key scaling for kk
        self.k_a = nn.Parameter(torch.empty(1, 1, C))  # key scaling by learning rate
        self.r_k = nn.Parameter(torch.empty(H, N))      # bonus term weight

        # Time shift via zero-padding (shifts sequence by 1 position)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # Linear projections
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        # GroupNorm after WKV (one group per head, eps=64e-5 per reference)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize following the RWKV-7 training code conventions."""
        C = self.config.hidden_size
        H = self.n_head
        N = self.head_size
        n_layer = self.config.num_layers
        layer_id = self.layer_id

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

            ddd = torch.arange(C, dtype=torch.float32) / C

            # Token-shift lerp init (from reference training code)
            self.x_r.data = 1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0).reshape(1, 1, C)
            self.x_w.data = 1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0).reshape(1, 1, C)
            self.x_k.data = 1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0).reshape(1, 1, C)
            self.x_v.data = 1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0).reshape(1, 1, C)
            self.x_a.data = 1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0).reshape(1, 1, C)
            self.x_g.data = 1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0).reshape(1, 1, C)

            # Zigzag and linear patterns for init
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])

            # Decay w0 init
            www = torch.zeros(C)
            for n in range(C):
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
            self.w0.data = (www + 0.5 + zigzag * 2.5).reshape(1, 1, C)

            # LoRA init: zero first matrix, orthogonal second (small scale)
            def ortho_init(x, scale):
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                return x

            nn.init.zeros_(self.w1)
            ortho_init(self.w2, 0.1)

            # In-context learning rate a
            self.a0.data = (torch.zeros(C) - 0.19 + zigzag * 0.3 + linear * 0.4).reshape(1, 1, C)
            nn.init.zeros_(self.a1)
            ortho_init(self.a2, 0.1)

            # Value residual v
            self.v0.data = (torch.zeros(C) + 0.73 - linear * 0.4).reshape(1, 1, C)
            nn.init.zeros_(self.v1)
            ortho_init(self.v2, 0.1)

            # Gate g
            nn.init.zeros_(self.g1)
            ortho_init(self.g2, 0.1)

            # Key normalization / scaling
            self.k_k.data = (torch.zeros(C) + 0.71 - linear * 0.1).reshape(1, 1, C)
            self.k_a.data = (torch.zeros(C) + 1.02).reshape(1, 1, C)
            self.r_k.data = torch.zeros(H, N) - 0.04

            # Linear projections
            c_sqrt = C ** 0.5
            self.receptance.weight.data.uniform_(-0.5 / c_sqrt, 0.5 / c_sqrt)
            self.key.weight.data.uniform_(-0.05 / c_sqrt, 0.05 / c_sqrt)
            self.value.weight.data.uniform_(-0.5 / c_sqrt, 0.5 / c_sqrt)
            nn.init.zeros_(self.output.weight)

    def forward(self, x, v_first):
        """
        Args:
            x: (B, T, C) input (after LayerNorm)
            v_first: (B, T, C) value from layer 0, or empty tensor as placeholder

        Returns:
            out: (B, T, C)
            v_first: (B, T, C) updated (set on layer 0, passed through on others)
        """
        B, T, C = x.size()
        H = self.n_head
        N = self.head_size

        # Token shift: xx = x_prev - x
        xx = self.time_shift(x) - x

        # Compute shifted inputs for each branch
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # Linear projections
        r = self.receptance(xr)
        w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2  # raw w (log-domain processed in attn)
        k = self.key(xk)
        v = self.value(xv)

        # Value residual: layer 0 stores v_first, later layers blend
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # In-context learning rate
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)

        # Gate
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        # Normalized key for delta-rule update
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)

        # Scale key by learning rate
        k = k * (1 + (a - 1) * self.k_a)

        # WKV computation: state = state * w + state @ (-kk @ (kk*a)^T) + v @ k^T
        # The a/b args to the attn function are the removal/replacement keys
        x_out = rwkv7_attn_parallel(r, w, k, v, -kk, kk * a, N)

        # GroupNorm
        x_out = self.ln_x(x_out.view(B * T, C)).view(B, T, C)

        # Bonus term: sum(r * k * r_k, dim=-1) * v  (per head)
        bonus = (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
            .sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)
        ).view(B, T, C)
        x_out = x_out + bonus

        # Output projection with gate
        x_out = self.output(x_out * g)
        return x_out, v_first

    def forward_rnn(self, x, x_prev, v_first, state):
        """
        Single-token RNN mode for autoregressive generation.

        Args:
            x: (B, C) current token embedding (after LayerNorm)
            x_prev: (B, C) previous token embedding
            v_first: (B, C) value from layer 0
            state: (B, H, N, N) recurrent state

        Returns:
            out: (B, C)
            x (for next step's x_prev): (B, C)
            v_first: (B, C)
            state: (B, H, N, N)
        """
        B, C = x.size()
        H = self.n_head
        N = self.head_size

        xx = x_prev - x

        xr = x + xx * self.x_r.squeeze(0)
        xw = x + xx * self.x_w.squeeze(0)
        xk = x + xx * self.x_k.squeeze(0)
        xv = x + xx * self.x_v.squeeze(0)
        xa = x + xx * self.x_a.squeeze(0)
        xg = x + xx * self.x_g.squeeze(0)

        r = self.receptance(xr)
        w = self.w0.squeeze(0) + torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0.squeeze(0) + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0.squeeze(0) + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k.squeeze(0)
        kk = F.normalize(kk.view(B, H, N), dim=-1, p=2.0).view(B, C)
        k = k * (1 + (a - 1) * self.k_a.squeeze(0))

        # Single-step WKV
        x_out, state = rwkv7_attn_sequential(
            r.unsqueeze(1), w.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
            (-kk).unsqueeze(1), (kk * a).unsqueeze(1),
            state, N
        )
        x_out = x_out.squeeze(1)

        # GroupNorm
        x_out = self.ln_x(x_out)

        # Bonus term
        bonus = (
            (r.view(B, H, N) * k.view(B, H, N) * self.r_k)
            .sum(dim=-1, keepdim=True) * v.view(B, H, N)
        ).view(B, C)
        x_out = x_out + bonus

        x_out = self.output(x_out * g)
        return x_out, x, v_first, state


# ---------------------------------------------------------------------------
# RWKV-7 Channel Mixing (FFN)
# ---------------------------------------------------------------------------

class RWKV_ChannelMix(nn.Module):
    """
    RWKV-7 Channel Mixing: token-shift + squared-ReLU FFN.

    This is a simple gated FFN where:
        k = x + (x_prev - x) * mu_k
        out = value(relu(key(k))^2)
    """

    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        C = config.hidden_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))

        self.key = nn.Linear(C, config.dim_ffn, bias=False)
        self.value = nn.Linear(config.dim_ffn, C, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        C = self.config.hidden_size

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (self.layer_id / self.config.num_layers)
            ddd = torch.arange(C, dtype=torch.float32) / C
            self.x_k.data = 1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0).reshape(1, 1, C)

            # Orthogonal init for key, zero for value (per reference)
            nn.init.orthogonal_(self.key.weight, gain=2.0)
            nn.init.zeros_(self.value.weight)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            out: (B, T, C)
        """
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

    def forward_rnn(self, x, x_prev):
        """
        Single-token RNN mode.

        Args:
            x: (B, C)
            x_prev: (B, C)
        Returns:
            out: (B, C)
            x: (B, C) -- for next step
        """
        xx = x_prev - x
        k = x + xx * self.x_k.squeeze(0)
        k = torch.relu(self.key(k)) ** 2
        return self.value(k), x


# ---------------------------------------------------------------------------
# RWKV-7 Block
# ---------------------------------------------------------------------------

class RWKVBlock(nn.Module):
    """
    Single RWKV-7 block: LayerNorm -> TimeMix -> residual -> LayerNorm -> ChannelMix -> residual

    Block 0 has an extra ln0 applied to the raw embeddings (pre-LN for the embedding).
    """

    def __init__(self, config: RWKVConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        C = config.hidden_size

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(C)

        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

        self.att = RWKV_TimeMix(config, layer_id)
        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        att_out, v_first = self.att(self.ln1(x), v_first)
        x = x + att_out
        x = x + self.ffn(self.ln2(x))
        return x, v_first


# ---------------------------------------------------------------------------
# RWKV-7 Model
# ---------------------------------------------------------------------------

class RWKVModel(nn.Module):
    """
    Complete RWKV-7 "Goose" language model.

    Architecture: Embedding -> [Block x num_layers] -> LayerNorm -> LM Head

    Supports:
    - Parallel training mode (full sequence processing)
    - RNN inference mode (token-by-token generation)
    - Gradient checkpointing
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            RWKVBlock(config, i) for i in range(config.num_layers)
        ])
        self.ln_out = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._gradient_checkpointing = config.gradient_checkpointing

        # Apply custom init for embedding and head
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.emb.weight.data.normal_(mean=0.0, std=0.02)
            # Head is NOT tied to embedding in RWKV-7
            self.head.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None):
        """
        Forward pass in parallel (training) mode.

        Args:
            input_ids: (B, T) token indices
            labels: (B, T) target token indices (optional, for loss)
            attention_mask: (B, T) mask (optional, for padding -- 1=keep, 0=ignore)

        Returns:
            dict with "loss", "logits", "aux_loss"
        """
        x = self.emb(input_ids)
        v_first = torch.empty_like(x)

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x, v_first = torch.utils.checkpoint.checkpoint(
                    block, x, v_first, use_reentrant=False
                )
            else:
                x, v_first = block(x, v_first)

        logits = self.head(self.ln_out(x))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # If attention_mask is provided, mask out padding positions
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                # Flatten
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_mask = shift_mask.view(-1)

                # Set labels to -100 where mask is 0 (padding)
                shift_labels = shift_labels.clone()
                shift_labels[shift_mask == 0] = -100

                loss = F.cross_entropy(
                    shift_logits, shift_labels, ignore_index=-100
                )
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": torch.tensor(0.0, device=input_ids.device, dtype=logits.dtype),
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def _init_rnn_state(self, batch_size, device, dtype):
        """Initialize RNN state for autoregressive generation."""
        C = self.config.hidden_size
        H = self.config.num_heads
        N = self.config.head_size
        n_layer = self.config.num_layers

        # Per layer: att_x_prev (B, C), att_state (B, H, N, N), ffn_x_prev (B, C)
        att_x_prev = [torch.zeros(batch_size, C, device=device, dtype=dtype) for _ in range(n_layer)]
        att_state = [torch.zeros(batch_size, H, N, N, device=device, dtype=torch.float32) for _ in range(n_layer)]
        ffn_x_prev = [torch.zeros(batch_size, C, device=device, dtype=dtype) for _ in range(n_layer)]
        v_first = [torch.zeros(batch_size, C, device=device, dtype=dtype)]

        return att_x_prev, att_state, ffn_x_prev, v_first

    def _forward_rnn_one_token(self, token_id, att_x_prev, att_state, ffn_x_prev, v_first_list):
        """Process a single token in RNN mode."""
        x = self.emb(token_id)  # (B, C) -- token_id is (B,)

        v_first = v_first_list[0]

        for i, block in enumerate(self.blocks):
            if i == 0:
                x = block.ln0(x)

            xx = block.ln1(x)
            att_out, new_x_prev, new_v_first, new_att_state = block.att.forward_rnn(
                xx, att_x_prev[i], v_first, att_state[i]
            )
            att_x_prev[i] = new_x_prev
            att_state[i] = new_att_state
            if i == 0:
                v_first = new_v_first
            x = x + att_out

            xx = block.ln2(x)
            ffn_out, new_ffn_x = block.ffn.forward_rnn(xx, ffn_x_prev[i])
            ffn_x_prev[i] = new_ffn_x
            x = x + ffn_out

        v_first_list[0] = v_first

        logits = self.head(self.ln_out(x))  # (B, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=256,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=2,
    ):
        """
        Autoregressive generation using RNN mode for O(1) per-token inference.

        For the prompt, we process tokens one at a time to build up the RNN state.
        Then we generate new tokens autoregressively.

        Args:
            input_ids: (1, T) or (B, T) prompt token IDs
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            repetition_penalty: penalty for repeated tokens
            eos_token_id: end-of-sequence token ID

        Returns:
            input_ids: (B, T + generated) full sequence including prompt
        """
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device
        dtype = self.emb.weight.dtype

        att_x_prev, att_state, ffn_x_prev, v_first = self._init_rnn_state(B, device, dtype)

        # Process prompt tokens to build up state
        for t in range(T):
            logits = self._forward_rnn_one_token(
                input_ids[:, t], att_x_prev, att_state, ffn_x_prev, v_first
            )

        # Generate new tokens
        generated = []
        for _ in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for tid in set(input_ids[b].tolist() + [g.item() for g in generated]):
                        logits[b, tid] /= repetition_penalty

            # Temperature
            logits_scaled = logits / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits_scaled, min(top_k, logits_scaled.size(-1)))
                logits_scaled[logits_scaled < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_scaled, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                logits_scaled[remove_mask.scatter(1, sorted_indices, remove_mask)] = float("-inf")

            # Sample
            probs = F.softmax(logits_scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated.append(next_token.squeeze(-1))

            # Check EOS
            if B == 1 and next_token.item() == eos_token_id:
                break

            # Feed next token through RNN
            logits = self._forward_rnn_one_token(
                next_token.squeeze(-1), att_x_prev, att_state, ffn_x_prev, v_first
            )

        if generated:
            generated_tensor = torch.stack(generated, dim=1)  # (B, num_generated)
            input_ids = torch.cat([input_ids, generated_tensor], dim=1)

        return input_ids


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("RWKV-7 'Goose' Model -- Standalone Test")
    print("=" * 70)

    config = RWKVConfig()

    print(f"\nConfig:")
    print(f"  vocab_size     = {config.vocab_size}")
    print(f"  hidden_size    = {config.hidden_size}")
    print(f"  num_layers     = {config.num_layers}")
    print(f"  num_heads      = {config.num_heads}")
    print(f"  head_size      = {config.head_size}")
    print(f"  dim_ffn        = {config.dim_ffn}")
    print(f"  max_seq_len    = {config.max_seq_len}")
    print(f"  Param estimate = {config.total_params_estimate():,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    model = RWKVModel(config).to(device)
    total, trainable = model.count_parameters()
    print(f"Actual parameters: {total:,} total, {trainable:,} trainable")
    print(f"Model size (fp32): {total * 4 / 1e9:.3f} GB")
    print(f"Model size (fp16): {total * 2 / 1e9:.3f} GB")

    # Test forward pass
    print("\n--- Forward Pass Test ---")
    B, T = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Disable gradient checkpointing for test
    model._gradient_checkpointing = False
    model.train()

    outputs = model(input_ids, labels=labels)
    print(f"  Input shape:  {input_ids.shape}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss:         {outputs['loss'].item():.4f}")
    print(f"  Aux loss:     {outputs['aux_loss'].item():.4f}")

    # Test backward pass
    print("\n--- Backward Pass Test ---")
    outputs["loss"].backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  Gradient norm: {grad_norm:.4f}")

    # Test generation (RNN mode)
    print("\n--- Generation Test (RNN mode) ---")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=16, temperature=1.0, top_k=50)
    print(f"  Prompt length:    {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  Generated tokens: {generated[0].tolist()}")

    # Test with attention_mask
    print("\n--- Attention Mask Test ---")
    mask = torch.ones(B, T, device=device, dtype=torch.long)
    mask[:, -10:] = 0  # mask last 10 positions
    outputs_masked = model(input_ids, labels=labels, attention_mask=mask)
    print(f"  Masked loss: {outputs_masked['loss'].item():.4f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

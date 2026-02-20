"""
Alternative Architecture Implementations for Tournament Training
================================================================
All models inherit BaseLanguageModel from model.py.
Block interface: forward(x) -> x  (residual included in block).

Architectures:
  1. Mamba       - Selective State Space Model (Gu & Dao, 2023)
  2. RWKV        - Linear attention RNN with exponential decay
  3. xLSTM       - Extended LSTM with matrix memory (Beck et al., 2024)
  4. Hawk        - RG-LRU gated linear recurrence (De et al., 2024)
  5. Griffin     - Hawk + local attention hybrid (De et al., 2024)
  6. RetNet      - Multi-scale retention network (Sun et al., 2023)
  7. Liquid      - Input-dependent SSM (Liquid Time-Constant Networks)
  8. MambaAttn   - Mamba-Attention hybrid (Jamba-style)

Note: All recurrent scans use sequential (loop-over-timesteps) for
correctness during tournament. If a variant wins, optimize later.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import (
    HamnerConfig, BaseLanguageModel, RMSNorm,
    precompute_rope, apply_rope,
)


# ============================================================================
# Shared components
# ============================================================================

class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward used by architectures needing a separate FFN."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LocalAttentionBlock(nn.Module):
    """Self-contained causal attention block with its own RoPE buffers.
    Same forward(x)->x interface as all other blocks."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = hidden // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.ln1 = RMSNorm(hidden)
        self.ln2 = RMSNorm(hidden)

        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

        rope_cos, rope_sin = precompute_rope(
            self.head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.mlp = SwiGLUMLP(hidden, config.expert_intermediate_size)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)

        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) + causal
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)

        x = x + self.o_proj(out)
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================================
# Scan helpers — excluded from torch.compile (Python for-loops are too slow
# to trace). The projections/norms before and after still get compiled.
# ============================================================================

@torch.compiler.disable
def _mamba_scan(x_ssm, delta, A, B_input, C_input, d_inner, d_state):
    batch, seq_len, _ = x_ssm.shape
    h = torch.zeros(batch, d_inner, d_state, device=x_ssm.device, dtype=x_ssm.dtype)
    outputs = []
    for t in range(seq_len):
        dt = delta[:, t, :, None]
        dA = torch.exp(dt * A.unsqueeze(0))
        dB = dt * B_input[:, t, None, :]
        h = dA * h + dB * x_ssm[:, t, :, None]
        outputs.append((h * C_input[:, t, None, :]).sum(-1))
    return torch.stack(outputs, dim=1)


@torch.compiler.disable
def _rwkv_wkv_scan(k, v, w, u, B, T, C, dtype):
    a = torch.zeros(B, C, device=k.device, dtype=torch.float32)
    b = torch.zeros(B, C, device=k.device, dtype=torch.float32)
    outputs = []
    for t in range(T):
        kt = k[:, t].float()
        vt = v[:, t].float()
        ekt = torch.exp(torch.clamp(u + kt, -20, 20))
        wkv = (a + ekt * vt) / (b + ekt + 1e-8)
        outputs.append(wkv.to(dtype))
        ew = torch.exp(torch.clamp(w, -20, 20))
        ek = torch.exp(torch.clamp(kt, -20, 20))
        a = ew * a + ek * vt
        b = ew * b + ek
    return torch.stack(outputs, dim=1)


@torch.compiler.disable
def _xlstm_scan(q, k, v, i_gate, f_gate, B, T, H, d, dtype):
    C = torch.zeros(B, H, d, d, device=q.device, dtype=dtype)
    n = torch.zeros(B, H, d, device=q.device, dtype=dtype)
    outputs = []
    for t in range(T):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        it = i_gate[:, t, :, None, None]
        ft = f_gate[:, t, :, None, None]
        C = ft * C + it * torch.einsum('bhd,bhe->bhde', vt, kt)
        n = ft.squeeze(-1) * n + i_gate[:, t, :, None] * kt
        ht = torch.einsum('bhde,bhe->bhd', C, qt)
        denom = torch.clamp(
            torch.einsum('bhd,bhd->bh', n, qt).abs(), min=1.0
        ).unsqueeze(-1)
        outputs.append(ht / denom)
    return torch.stack(outputs, dim=1)


@torch.compiler.disable
def _hawk_scan(x_in, a, d_recurrent, B, T, dtype):
    state = torch.zeros(B, d_recurrent, device=x_in.device, dtype=dtype)
    outputs = []
    for t in range(T):
        at = a[:, t]
        state = at * state + torch.sqrt(1 - at * at + 1e-8) * x_in[:, t]
        outputs.append(state)
    return torch.stack(outputs, dim=1)


@torch.compiler.disable
def _retnet_scan(q, k, v, gamma, B, T, H, d, dtype):
    S = torch.zeros(B, H, d, d, device=q.device, dtype=dtype)
    outputs = []
    for t in range(T):
        S = gamma[None, :, None, None] * S + \
            torch.einsum('bhd,bhe->bhde', k[:, t], v[:, t])
        outputs.append(torch.einsum('bhd,bhde->bhe', q[:, t], S))
    return torch.stack(outputs, dim=1)


@torch.compiler.disable
def _liquid_scan(x_in, a_mod, a_base, B_input, C_input, d_inner, d_state, B, T, dtype):
    h = torch.zeros(B, d_inner, d_state, device=x_in.device, dtype=dtype)
    outputs = []
    for t in range(T):
        at = a_mod[:, t, :, None] * a_base[None, :]
        bt = B_input[:, t, None, :]
        h = at * h + bt * x_in[:, t, :, None]
        outputs.append((h * C_input[:, t, None, :]).sum(-1))
    return torch.stack(outputs, dim=1)


# ============================================================================
# 1. Mamba (Selective State Space Model)
# ============================================================================

class MambaBlock(nn.Module):
    """Selective SSM block: expand -> conv1d -> selective scan -> gate -> project."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.d_inner = config.hidden_size * config.expand_factor
        self.d_state = config.d_state
        self.d_conv = config.d_conv

        self.norm = RMSNorm(config.hidden_size)

        # Input projection: x -> (x_ssm, z_gate)
        self.in_proj = nn.Linear(config.hidden_size, self.d_inner * 2, bias=False)

        # Depthwise conv on the SSM path
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=config.d_conv, padding=config.d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # Input-dependent SSM parameters: B, C, delta
        self.x_proj = nn.Linear(
            self.d_inner, config.d_state * 2 + self.d_inner, bias=False
        )

        # A is a fixed log-space diagonal
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, config.hidden_size, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Conv1d path
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        # Input-dependent SSM params
        proj = self.x_proj(x_ssm)
        B_input = proj[..., :self.d_state]
        C_input = proj[..., self.d_state:2 * self.d_state]
        delta = F.softplus(proj[..., 2 * self.d_state:])  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), negative

        y = _mamba_scan(x_ssm, delta, A, B_input, C_input, self.d_inner, self.d_state)
        y = (y + self.D * x_ssm) * F.silu(z)
        return residual + self.out_proj(y)


class MambaModel(BaseLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            MambaBlock(config, i) for i in range(config.num_layers)
        ])
        self._post_init()


# ============================================================================
# 2. RWKV (Receptance Weighted Key Value)
# ============================================================================

class RWKVBlock(nn.Module):
    """RWKV block: time mixing (linear attn with decay) + channel mixing."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.expert_intermediate_size

        self.ln1 = RMSNorm(hidden)
        self.ln2 = RMSNorm(hidden)

        # --- Time mixing params ---
        self.time_decay = nn.Parameter(torch.randn(hidden) * 0.1 - 5.0)
        self.time_first = nn.Parameter(torch.randn(hidden) * 0.01)
        self.time_mix_k = nn.Parameter(torch.ones(hidden) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(hidden) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(hidden) * 0.5)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.receptance = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)

        # --- Channel mixing params ---
        self.time_mix_k2 = nn.Parameter(torch.ones(hidden) * 0.5)
        self.time_mix_r2 = nn.Parameter(torch.ones(hidden) * 0.5)
        self.key2 = nn.Linear(hidden, intermediate, bias=False)
        self.value2 = nn.Linear(intermediate, hidden, bias=False)
        self.receptance2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        zero = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)

        # --- Time Mixing ---
        h = self.ln1(x)
        shifted = torch.cat([zero, h[:, :-1]], dim=1)
        xk = h * self.time_mix_k + shifted * (1 - self.time_mix_k)
        xv = h * self.time_mix_v + shifted * (1 - self.time_mix_v)
        xr = h * self.time_mix_r + shifted * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        # WKV scan (float32 accumulation, clamped for stability)
        w = -torch.exp(self.time_decay)  # negative → exp(w) < 1 → decay
        u = self.time_first
        wkv_out = _rwkv_wkv_scan(k, v, w, u, B, T, C, x.dtype)

        x = x + self.output(r * wkv_out)

        # --- Channel Mixing ---
        h = self.ln2(x)
        shifted = torch.cat([zero, h[:, :-1]], dim=1)
        xk = h * self.time_mix_k2 + shifted * (1 - self.time_mix_k2)
        xr = h * self.time_mix_r2 + shifted * (1 - self.time_mix_r2)
        k2 = torch.relu(self.key2(xk)) ** 2   # squared ReLU
        r2 = torch.sigmoid(self.receptance2(xr))
        x = x + r2 * self.value2(k2)

        return x


class RWKVModel(BaseLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            RWKVBlock(config, i) for i in range(config.num_layers)
        ])
        self._post_init()


# ============================================================================
# 3. xLSTM (Extended LSTM with Matrix Memory)
# ============================================================================

class XLSTMBlock(nn.Module):
    """mLSTM block: multi-head matrix-memory LSTM + SwiGLU MLP.
    Memory C is (H, d, d) per head — stores key-value associations.
    Exponential input gate + sigmoid forget gate.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads

        self.ln1 = RMSNorm(hidden)
        self.ln2 = RMSNorm(hidden)

        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

        # Per-head gates
        self.i_proj = nn.Linear(hidden, self.num_heads, bias=True)
        self.f_proj = nn.Linear(hidden, self.num_heads, bias=True)
        nn.init.constant_(self.f_proj.bias, 3.0)  # start with high forget = long memory

        self.mlp = SwiGLUMLP(hidden, config.expert_intermediate_size)

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, H, d)
        k = self.k_proj(h).view(B, T, H, d)
        v = self.v_proj(h).view(B, T, H, d)

        i_gate = torch.exp(torch.clamp(self.i_proj(h), -10, 10))  # (B, T, H)
        f_gate = torch.sigmoid(self.f_proj(h))                     # (B, T, H)

        # Sequential mLSTM scan
        y = _xlstm_scan(q, k, v, i_gate, f_gate, B, T, H, d, x.dtype).reshape(B, T, D)
        x = x + self.o_proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class XLSTMModel(BaseLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            XLSTMBlock(config, i) for i in range(config.num_layers)
        ])
        self._post_init()


# ============================================================================
# 4. Hawk (RG-LRU: Real-Gated Linear Recurrence Unit)
# ============================================================================

class HawkBlock(nn.Module):
    """RG-LRU: h_t = a_t * h_{t-1} + sqrt(1-a_t^2) * x_t
    With input-dependent gating and SwiGLU MLP.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden = config.hidden_size
        self.d_recurrent = hidden * config.expand_factor

        self.ln1 = RMSNorm(hidden)
        self.ln2 = RMSNorm(hidden)

        self.input_proj = nn.Linear(hidden, self.d_recurrent, bias=False)
        self.gate_proj = nn.Linear(hidden, self.d_recurrent, bias=False)

        # Recurrence gate: base + input-dependent
        self.a_param = nn.Parameter(torch.randn(self.d_recurrent) * 0.1)
        self.a_input = nn.Linear(self.d_recurrent, self.d_recurrent, bias=False)

        self.output_proj = nn.Linear(self.d_recurrent, hidden, bias=False)
        self.mlp = SwiGLUMLP(hidden, config.expert_intermediate_size)

    def forward(self, x):
        B, T, D = x.shape

        h = self.ln1(x)
        x_in = self.input_proj(h)
        gate = torch.sigmoid(self.gate_proj(h))

        a = torch.sigmoid(self.a_param + self.a_input(x_in))  # (B, T, d_rec)

        y = gate * _hawk_scan(x_in, a, self.d_recurrent, B, T, x.dtype)
        x = x + self.output_proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class HawkModel(BaseLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            HawkBlock(config, i) for i in range(config.num_layers)
        ])
        self._post_init()


# ============================================================================
# 5. Griffin (Hawk + Local Attention Hybrid)
# ============================================================================

class GriffinModel(BaseLanguageModel):
    """Hawk blocks with local attention every N layers."""
    def __init__(self, config):
        super().__init__(config)
        blocks = []
        for i in range(config.num_layers):
            if (i + 1) % config.attn_every_n == 0:
                blocks.append(LocalAttentionBlock(config, i))
            else:
                blocks.append(HawkBlock(config, i))
        self.blocks = nn.ModuleList(blocks)
        self._post_init()


# ============================================================================
# 6. RetNet (Retentive Network with Multi-Scale Retention)
# ============================================================================

class RetNetBlock(nn.Module):
    """Multi-scale retention: like attention but with exponential decay, no softmax.
    Each head has its own decay rate (multi-scale).
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads

        self.ln1 = RMSNorm(hidden)
        self.ln2 = RMSNorm(hidden)

        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.g_proj = nn.Linear(hidden, hidden, bias=False)  # output gate
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

        # Per-head decay (multi-scale): spread from ~0.9 to ~0.999
        gammas = 1.0 - torch.exp(
            torch.linspace(math.log(0.1), math.log(0.001), self.num_heads)
        )
        self.gamma = nn.Parameter(gammas)

        self.group_norm = nn.GroupNorm(self.num_heads, hidden)
        self.mlp = SwiGLUMLP(hidden, config.expert_intermediate_size)

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, H, d)
        k = self.k_proj(h).view(B, T, H, d)
        v = self.v_proj(h).view(B, T, H, d)
        g = torch.sigmoid(self.g_proj(h))

        gamma = torch.sigmoid(self.gamma)  # (H,) in (0, 1)

        # Sequential retention scan: S = gamma * S + k v^T; y = q @ S
        y = _retnet_scan(q, k, v, gamma, B, T, H, d, x.dtype).reshape(B, T, D)
        y = self.group_norm(y.transpose(1, 2)).transpose(1, 2)
        y = g * y
        x = x + self.o_proj(y)

        x = x + self.mlp(self.ln2(x))
        return x


class RetNetModel(BaseLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            RetNetBlock(config, i) for i in range(config.num_layers)
        ])
        self._post_init()


# ============================================================================
# 7. Liquid (Input-Dependent State Space Model)
# ============================================================================

class LiquidBlock(nn.Module):
    """Liquid Neural Network: state transition A depends on input.
    Unlike Mamba (fixed A, input-dependent discretization), Liquid
    makes A itself input-dependent — dynamics adapt to content.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        hidden = config.hidden_size
        self.d_inner = hidden * config.expand_factor
        self.d_state = config.d_state

        self.norm = RMSNorm(hidden)

        self.in_proj = nn.Linear(hidden, self.d_inner * 2, bias=False)

        # Input-dependent A: projects to per-dim gate that modulates base decay
        self.A_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # Multi-scale base decay per state dim
        A_base = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        self.A_base = nn.Parameter(torch.log(A_base))

        self.B_proj = nn.Linear(self.d_inner, config.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, config.d_state, bias=False)

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, hidden, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = F.silu(x_in)

        # Input-dependent decay gate
        a_mod = torch.sigmoid(self.A_proj(x_in))  # (B, T, d_inner)
        a_base = torch.sigmoid(self.A_base)        # (d_state,)
        B_input = self.B_proj(x_in)                # (B, T, d_state)
        C_input = self.C_proj(x_in)                # (B, T, d_state)

        # Sequential scan
        y = _liquid_scan(x_in, a_mod, a_base, B_input, C_input,
                         self.d_inner, self.d_state, B, T, x.dtype)
        y = (y + self.D * x_in) * F.silu(z)
        return residual + self.out_proj(y)


class LiquidModel(BaseLanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.blocks = nn.ModuleList([
            LiquidBlock(config, i) for i in range(config.num_layers)
        ])
        self._post_init()


# ============================================================================
# 8. Mamba-Attention Hybrid (Jamba-style)
# ============================================================================

class MambaAttentionModel(BaseLanguageModel):
    """Mostly Mamba SSM blocks with causal attention every N layers."""
    def __init__(self, config):
        super().__init__(config)
        blocks = []
        for i in range(config.num_layers):
            if (i + 1) % config.attn_every_n == 0:
                blocks.append(LocalAttentionBlock(config, i))
            else:
                blocks.append(MambaBlock(config, i))
        self.blocks = nn.ModuleList(blocks)
        self._post_init()

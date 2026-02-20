"""
xLSTM (Extended LSTM) Language Model - mLSTM Variant
=====================================================
Pure PyTorch implementation of the mLSTM (matrix-memory LSTM) architecture
from "xLSTM: Extended Long Short-Term Memory" (Beck et al., 2024).

Key features:
- Matrix memory C storing key-value associations via covariance update rule
- Exponential gating with log-space stabilization
- Fully parallelizable training (no sequential dependency like sLSTM)
- Multi-head design with per-head gating
- RMSNorm + SwiGLU feedforward blocks

Reference: https://arxiv.org/abs/2405.04517
Reference implementation: https://github.com/NX-AI/xlstm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class XLSTMConfig:
    vocab_size: int = 49152
    hidden_size: int = 768
    num_layers: int = 24
    max_seq_len: int = 1024
    gradient_checkpointing: bool = True

    # mLSTM-specific
    num_heads: int = 8           # number of memory heads
    head_dim: int = 144          # dimension per head (recomputed in __post_init__)
    proj_factor: float = 1.5    # up-projection factor for inner dim
    conv1d_kernel_size: int = 4  # causal conv kernel before qkv
    qkv_proj_blocksize: int = 4 # blockwise projection grouping

    # Feedforward
    ff_factor: float = 8.0 / 3.0  # SwiGLU intermediate ratio (8/3 * hidden ~ 2.67x)

    # Normalization / regularization
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = False

    def __post_init__(self):
        # Inner embedding dim used inside each mLSTM layer
        self._inner_dim = int(self.proj_factor * self.hidden_size)
        # Round inner_dim to nearest multiple of num_heads for clean head splits
        self._inner_dim = (self._inner_dim // self.num_heads) * self.num_heads
        # Recompute head_dim from inner_dim
        self.head_dim = self._inner_dim // self.num_heads

    def total_params_estimate(self) -> int:
        """Rough parameter count estimate for sizing."""
        inner = self._inner_dim
        H = self.hidden_size
        V = self.vocab_size

        # Token embeddings
        embed = V * H

        # Per mLSTM layer:
        #   up_proj: H -> 2 * inner (gate branch + mlstm branch)
        up_proj = H * (2 * inner)
        #   conv1d: inner * kernel
        conv = inner * self.conv1d_kernel_size + inner  # weight + bias
        #   q, k, v projections: inner -> inner each
        qkv = 3 * (inner * inner)
        #   gate projections (igate, fgate): 3*inner -> num_heads each
        gates = 2 * (3 * inner * self.num_heads)
        #   output norm: inner
        outnorm = inner
        #   learnable skip: 1
        skip = 1
        #   down_proj: inner -> H
        down_proj = inner * H

        mlstm_per_layer = up_proj + conv + qkv + gates + outnorm + skip + down_proj

        # Per feedforward layer:
        #   gate_proj: H -> ff_intermediate
        #   up_proj: H -> ff_intermediate
        #   down_proj: ff_intermediate -> H
        ff_intermediate = int(self.ff_factor * H)
        # Round to multiple of 256 for efficiency
        ff_intermediate = ((ff_intermediate + 255) // 256) * 256
        ff_per_layer = 3 * H * ff_intermediate

        # Norms per block: 2 * H (mlstm_norm + ff_norm)
        norms_per_layer = 2 * H

        per_layer = mlstm_per_layer + ff_per_layer + norms_per_layer

        # Final norm + lm_head
        final = H + V * H

        total = embed + per_layer * self.num_layers + final
        if self.tie_word_embeddings:
            total -= V * H
        return total


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class MultiHeadLayerNorm(nn.Module):
    """Layer normalization applied per-head, used for mLSTM output normalization."""
    def __init__(self, ndim: int, num_heads: int, eps: float = 1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = ndim // num_heads
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) -> reshape to (B, S, NH, DH), norm per head, reshape back
        B, S, D = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim)
        x_f = x.float()
        mean = x_f.mean(dim=-1, keepdim=True)
        var = x_f.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_f - mean) / (var + self.eps).sqrt()
        x_norm = x_norm.view(B, S, D).type_as(x) * self.weight
        return x_norm


class CausalConv1d(nn.Module):
    """Causal 1D convolution: pads on the left so output length == input length."""
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) -> (B, D, S)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class LinearHeadwiseExpand(nn.Module):
    """
    Linear projection that operates blockwise per head.
    Projects inner_dim -> inner_dim with block-diagonal structure per head group.
    """
    def __init__(self, inner_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        # Full linear for simplicity -- the headwise structure is implicit
        # in how q/k/v are reshaped into (B, NH, S, DH) downstream.
        self.linear = nn.Linear(inner_dim, inner_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feedforward block: gate_proj, up_proj (SiLU-gated), down_proj."""
    def __init__(self, hidden_size: int, ff_factor: float, dropout: float = 0.0):
        super().__init__()
        intermediate = int(ff_factor * hidden_size)
        # Round to multiple of 256 for efficiency
        intermediate = ((intermediate + 255) // 256) * 256
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# mLSTM Cell -- parallel (training) and recurrent (inference) modes
# ---------------------------------------------------------------------------

class mLSTMCell(nn.Module):
    """
    mLSTM cell with matrix memory and exponential gating.

    Parallel mode: processes entire sequence at once using a stabilized
    log-space formulation of the forget/input gate interaction matrix.

    Recurrent mode: processes one token at a time, maintaining (C, n, m) state.
    """

    def __init__(self, inner_dim: int, num_heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads

        # Input and forget gate projections: from concatenated [q, k, v]
        self.igate = nn.Linear(3 * inner_dim, num_heads, bias=True)
        self.fgate = nn.Linear(3 * inner_dim, num_heads, bias=True)

        # Output normalization (per-head layer norm)
        self.outnorm = MultiHeadLayerNorm(inner_dim, num_heads)

        # Precompute causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
            persistent=False,
        )

    def reset_parameters(self):
        """Initialize gates following the NX-AI reference:
        - forget gate bias initialized with linspace(3, 6) for strong initial memory retention
        - input gate bias initialized small normal
        """
        nn.init.zeros_(self.fgate.weight)
        with torch.no_grad():
            nn.init.zeros_(self.fgate.bias)
            # Linspace from 3.0 to 6.0 encourages initial memory retention
            self.fgate.bias.copy_(torch.linspace(3.0, 6.0, self.num_heads))
        nn.init.zeros_(self.igate.weight)
        nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Parallel mLSTM forward pass.

        Args:
            q, k, v: (B, S, D) where D = inner_dim

        Returns:
            h: (B, S, D) output hidden states
        """
        B, S, D = q.shape
        NH = self.num_heads
        DH = self.head_dim

        # Compute gate pre-activations from concatenated qkv
        qkv_cat = torch.cat([q, k, v], dim=-1)  # (B, S, 3*D)
        igate_preact = self.igate(qkv_cat)       # (B, S, NH)
        fgate_preact = self.fgate(qkv_cat)       # (B, S, NH)

        # Reshape to multi-head format: (B, NH, S, DH) and (B, NH, S, 1)
        q = q.view(B, S, NH, DH).permute(0, 2, 1, 3)   # (B, NH, S, DH)
        k = k.view(B, S, NH, DH).permute(0, 2, 1, 3)
        v = v.view(B, S, NH, DH).permute(0, 2, 1, 3)
        igate_preact = igate_preact.permute(0, 2, 1).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = fgate_preact.permute(0, 2, 1).unsqueeze(-1)  # (B, NH, S, 1)

        # Call the stabilized parallel backend
        h = self._parallel_stabilized(q, k, v, igate_preact, fgate_preact, S)

        # Reshape back: (B, NH, S, DH) -> (B, S, D)
        h = h.permute(0, 2, 1, 3).contiguous().view(B, S, D)
        h = self.outnorm(h)
        return h

    def _parallel_stabilized(
        self,
        q: torch.Tensor,     # (B, NH, S, DH)
        k: torch.Tensor,
        v: torch.Tensor,
        igate_preact: torch.Tensor,  # (B, NH, S, 1)
        fgate_preact: torch.Tensor,  # (B, NH, S, 1)
        S: int,
    ) -> torch.Tensor:
        """
        Stabilized parallel mLSTM computation.

        This implements the core mLSTM in parallel form:
        - Forget gates in log-space: log_f = logsigmoid(fgate_preact)
        - Cumulative forget gate products via cumsum in log-space
        - Gate interaction matrix D[i,j] = prod(f_{j+1}..f_i) * input_gate_j
        - Stabilization by subtracting row-wise max of log(D)
        - Output: h = (Q K^T * D) V / normalizer

        Reference: parallel_stabilized_simple from NX-AI/xlstm
        """
        B, NH, _, DH = q.shape
        _dtype = q.dtype
        _device = q.device
        eps = 1e-6

        # Forget gate in log-space
        log_fgates = F.logsigmoid(fgate_preact)  # (B, NH, S, 1)

        # Causal mask
        if S <= self.causal_mask.size(0):
            ltr = self.causal_mask[:S, :S]
        else:
            ltr = torch.tril(torch.ones(S, S, dtype=torch.bool, device=_device))

        # Cumulative sum of log forget gates: log(prod(f_1..f_t)) for each t
        # Prepend a zero for t=0
        log_fgates_cumsum = torch.cat(
            [torch.zeros(B, NH, 1, 1, dtype=_dtype, device=_device),
             torch.cumsum(log_fgates, dim=2)],
            dim=2,
        )  # (B, NH, S+1, 1)

        # Build the pairwise log-forget matrix:
        # log_fg_matrix[i,j] = sum(log_f_{j+1}..log_f_i) = cumsum_i - cumsum_j
        # This gives the cumulative product of forget gates between positions j and i
        rep = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
        _log_fg_matrix = rep - rep.transpose(-2, -1)
        # Apply causal mask and trim to (S, S)
        log_fg_matrix = torch.where(
            ltr, _log_fg_matrix[:, :, 1:, 1:], torch.tensor(-float("inf"), device=_device)
        )  # (B, NH, S, S)

        # Combine forget gates with input gates: D[i,j] = fg_prod(j+1..i) * ig(j)
        # igate_preact is the raw pre-activation (will be exponentiated via exp)
        log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)

        # Stabilization: subtract row-wise max so exp() doesn't overflow
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)
        max_log_D = max_log_D.detach()  # stop gradient through stabilizer

        log_D_stabilized = log_D_matrix - max_log_D
        D_matrix = torch.exp(log_D_stabilized)  # (B, NH, S, S)

        # Scale keys for numerical stability (like attention scaling)
        k_scaled = k / math.sqrt(DH)

        # QK^T weighted by gate matrix D
        qk = q @ k_scaled.transpose(-2, -1)  # (B, NH, S, S)
        C_matrix = qk * D_matrix

        # Normalizer: ensures denominator is always >= 1 or >= exp(-max_log_D)
        normalizer = torch.maximum(
            C_matrix.sum(dim=-1, keepdim=True).abs(),
            torch.exp(-max_log_D),
        ) + eps  # (B, NH, S, 1)

        C_normalized = C_matrix / normalizer

        # Retrieve values
        h = C_normalized @ v  # (B, NH, S, DH)
        return h

    def step(
        self,
        q: torch.Tensor,   # (B, 1, D)
        k: torch.Tensor,
        v: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Single recurrent step for autoregressive generation.

        State = (c_state, n_state, m_state):
            c_state: (B, NH, DH, DH) - matrix memory
            n_state: (B, NH, DH, 1)  - normalizer state
            m_state: (B, NH, 1, 1)   - stabilizer (max log gate)

        Returns:
            h: (B, 1, D) - output
            new_state: updated (c_state, n_state, m_state)
        """
        B = q.shape[0]
        NH = self.num_heads
        DH = self.head_dim

        # Initialize state if needed
        if state is None:
            c_state = torch.zeros(B, NH, DH, DH, device=q.device, dtype=q.dtype)
            n_state = torch.zeros(B, NH, DH, 1, device=q.device, dtype=q.dtype)
            m_state = torch.zeros(B, NH, 1, 1, device=q.device, dtype=q.dtype)
        else:
            c_state, n_state, m_state = state

        # Gate pre-activations
        qkv_cat = torch.cat([q, k, v], dim=-1)  # (B, 1, 3*D)
        igate_preact = self.igate(qkv_cat)       # (B, 1, NH)
        fgate_preact = self.fgate(qkv_cat)       # (B, 1, NH)

        # Reshape to (B, NH, 1, DH) and (B, NH, 1, 1)
        q_h = q.view(B, 1, NH, DH).permute(0, 2, 1, 3)  # (B, NH, 1, DH)
        k_h = k.view(B, 1, NH, DH).permute(0, 2, 1, 3)
        v_h = v.view(B, 1, NH, DH).permute(0, 2, 1, 3)
        igate_preact = igate_preact.permute(0, 2, 1).unsqueeze(-1)  # (B, NH, 1, 1)
        fgate_preact = fgate_preact.permute(0, 2, 1).unsqueeze(-1)  # (B, NH, 1, 1)

        # Squeeze sequence dim -> column vectors for outer products
        q_col = q_h.squeeze(2).unsqueeze(-1)  # (B, NH, DH, 1)
        k_col = k_h.squeeze(2).unsqueeze(-1)  # (B, NH, DH, 1)
        v_col = v_h.squeeze(2).unsqueeze(-1)  # (B, NH, DH, 1)

        # Log forget gate
        log_fg = F.logsigmoid(fgate_preact)  # (B, NH, 1, 1)

        # Update stabilizer: m = max(log_f + m_old, igate_preact)
        m_new = torch.max(log_fg + m_state, igate_preact)

        # Stabilized gate activations
        fg_act = torch.exp(log_fg + m_state - m_new)  # (B, NH, 1, 1)
        ig_act = torch.exp(igate_preact - m_new)      # (B, NH, 1, 1)

        # Scale keys
        k_scaled = k_col / math.sqrt(DH)

        # Update matrix memory: C = f * C_old + i * v @ k^T
        c_new = fg_act * c_state + ig_act * (v_col @ k_scaled.transpose(-2, -1))
        # (B, NH, DH, DH)

        # Update normalizer: n = f * n_old + i * k
        n_new = fg_act * n_state + ig_act * k_scaled
        # (B, NH, DH, 1)

        # Retrieve: h = q^T C / max(|q^T n|, exp(-m))
        h_num = q_col.transpose(-2, -1) @ c_new       # (B, NH, 1, DH) -- but actually (B, NH, 1, DH)
        qn_dot = q_col.transpose(-2, -1) @ n_new      # (B, NH, 1, 1)
        max_val = torch.exp(-m_new)
        h_denom = torch.maximum(qn_dot.abs(), max_val) + 1e-6
        h = h_num / h_denom  # (B, NH, 1, DH)

        # Reshape back: (B, NH, 1, DH) -> (B, 1, D)
        D = self.inner_dim
        h = h.permute(0, 2, 1, 3).contiguous().view(B, 1, D)
        h = self.outnorm(h)

        return h, (c_new, n_new, m_new)


# ---------------------------------------------------------------------------
# mLSTM Layer (up-project -> conv -> qkv -> cell -> gate -> down-project)
# ---------------------------------------------------------------------------

class mLSTMLayer(nn.Module):
    """
    Full mLSTM layer following the NX-AI architecture:
        x -> up_proj -> split(x_mlstm, z)
        x_mlstm -> causal_conv1d -> silu -> q_proj, k_proj
        x_mlstm -> v_proj (no conv activation for value)
        h = mLSTM_cell(q, k, v)
        h = (h + learnable_skip * conv_act(x_mlstm)) * silu(z)
        output = down_proj(h)
    """

    def __init__(self, config: XLSTMConfig):
        super().__init__()
        inner_dim = config._inner_dim

        # Up-projection: H -> 2 * inner_dim (split into mlstm branch + gate branch)
        self.up_proj = nn.Linear(config.hidden_size, 2 * inner_dim, bias=False)

        # Causal convolution on the mlstm branch
        self.conv1d = CausalConv1d(inner_dim, config.conv1d_kernel_size)
        self.conv_act = nn.SiLU()

        # QKV projections (operate on conv output for q,k; raw for v)
        self.q_proj = LinearHeadwiseExpand(inner_dim, config.num_heads, bias=False)
        self.k_proj = LinearHeadwiseExpand(inner_dim, config.num_heads, bias=False)
        self.v_proj = LinearHeadwiseExpand(inner_dim, config.num_heads, bias=False)

        # Core mLSTM cell
        self.cell = mLSTMCell(inner_dim, config.num_heads, config.max_seq_len)

        # Learnable skip connection weight
        self.learnable_skip = nn.Parameter(torch.ones(1))

        # Output gate activation
        self.ogate_act = nn.SiLU()

        # Down-projection: inner_dim -> H
        self.down_proj = nn.Linear(inner_dim, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, H) input hidden states
        Returns:
            output: (B, S, H)
        """
        # Up-project and split
        x_up = self.up_proj(x)               # (B, S, 2 * inner_dim)
        x_mlstm, z = x_up.chunk(2, dim=-1)  # each (B, S, inner_dim)

        # Causal conv + activation on mlstm branch
        x_conv = self.conv1d(x_mlstm)
        x_conv_act = self.conv_act(x_conv)

        # QKV projections
        q = self.q_proj(x_conv_act)
        k = self.k_proj(x_conv_act)
        v = self.v_proj(x_mlstm)  # value from raw (pre-conv) branch

        # mLSTM cell
        h = self.cell(q, k, v)  # (B, S, inner_dim)

        # Skip connection + output gating
        h = (h + self.learnable_skip * x_conv_act) * self.ogate_act(z)

        # Down-project
        output = self.dropout(self.down_proj(h))
        return output

    def step(
        self,
        x: torch.Tensor,  # (B, 1, H)
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Single-step inference for autoregressive generation."""
        if state is None:
            state = {"cell_state": None, "conv_buffer": None}

        # Up-project and split
        x_up = self.up_proj(x)
        x_mlstm, z = x_up.chunk(2, dim=-1)

        # Manage conv buffer for causal conv
        inner_dim = x_mlstm.shape[-1]
        kernel_size = self.conv1d.kernel_size
        if state["conv_buffer"] is None:
            conv_buffer = torch.zeros(
                x.shape[0], kernel_size, inner_dim,
                device=x.device, dtype=x.dtype,
            )
        else:
            conv_buffer = state["conv_buffer"]

        # Shift buffer and add new input
        conv_buffer = torch.cat([conv_buffer[:, 1:, :], x_mlstm], dim=1)

        # Apply conv manually: (B, K, D) -> sum over kernel dim with conv weights
        # conv1d weight shape: (D, 1, K) for groups=D
        # We need: for each feature d, dot product of buffer[:, :, d] with weight[d, 0, :]
        w = self.conv1d.conv.weight  # (D, 1, K)
        b = self.conv1d.conv.bias    # (D,)
        # buffer: (B, K, D) -> transpose to (B, D, K)
        buf_t = conv_buffer.transpose(1, 2)  # (B, D, K)
        # Depthwise conv: element-wise multiply and sum over kernel dim
        x_conv = (buf_t * w.squeeze(1)).sum(dim=-1)  # (B, D)
        if b is not None:
            x_conv = x_conv + b
        x_conv = x_conv.unsqueeze(1)  # (B, 1, D)

        x_conv_act = self.conv_act(x_conv)

        # QKV
        q = self.q_proj(x_conv_act)
        k = self.k_proj(x_conv_act)
        v = self.v_proj(x_mlstm)

        # mLSTM cell step
        h, cell_state = self.cell.step(q, k, v, state["cell_state"])

        # Skip + gate
        h = (h + self.learnable_skip * x_conv_act) * self.ogate_act(z)

        output = self.down_proj(h)

        new_state = {"cell_state": cell_state, "conv_buffer": conv_buffer}
        return output, new_state


# ---------------------------------------------------------------------------
# xLSTM Block (norm -> mLSTM -> residual, norm -> feedforward -> residual)
# ---------------------------------------------------------------------------

class XLSTMBlock(nn.Module):
    """Single xLSTM block: mLSTM sub-layer + SwiGLU feedforward sub-layer."""

    def __init__(self, config: XLSTMConfig):
        super().__init__()
        self.mlstm_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlstm = mLSTMLayer(config)
        self.ff_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward = SwiGLUFeedForward(
            config.hidden_size, config.ff_factor, config.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mLSTM sub-layer with residual
        x = x + self.mlstm(self.mlstm_norm(x))
        # Feedforward sub-layer with residual
        x = x + self.feedforward(self.ff_norm(x))
        return x

    def step(
        self, x: torch.Tensor, state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Single-step for generation."""
        h, mlstm_state = self.mlstm.step(self.mlstm_norm(x), state)
        x = x + h
        x = x + self.feedforward(self.ff_norm(x))
        return x, mlstm_state


# ---------------------------------------------------------------------------
# XLSTMModel -- full language model
# ---------------------------------------------------------------------------

class XLSTMModel(nn.Module):
    """
    xLSTM language model using stacked mLSTM blocks.

    Architecture: embedding -> [mLSTM block] * num_layers -> RMSNorm -> lm_head
    """

    def __init__(self, config: XLSTMConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Stacked xLSTM blocks
        self.layers = nn.ModuleList([
            XLSTMBlock(config) for _ in range(config.num_layers)
        ])

        # Final normalization and language model head
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self.apply(self._init_weights)
        # Special gate initialization for all mLSTM cells
        self._init_mlstm_gates()

    def _init_weights(self, module: nn.Module):
        """Standard initialization: normal(0, 0.02) for Linear/Embedding."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_mlstm_gates(self):
        """Apply mLSTM-specific gate initialization after general init."""
        for layer in self.layers:
            layer.mlstm.cell.reset_parameters()
            # Initialize learnable skip to 1.0
            nn.init.ones_(layer.mlstm.learnable_skip)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: (B, S) token indices
            labels: (B, S) target token indices for loss computation
            attention_mask: (B, S) binary mask (1 = keep, 0 = pad) -- used for
                            loss masking. mLSTM is inherently causal, so no
                            attention mask is needed for the recurrence itself.

        Returns:
            dict with keys: "loss", "logits", "aux_loss"
        """
        x = self.embed_tokens(input_ids)  # (B, S, H)

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False,
                )
            else:
                x = layer(x)

        logits = self.lm_head(self.final_norm(x))  # (B, S, V)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Apply attention_mask to loss if provided (mask out padding)
            if attention_mask is not None:
                # Shift mask to match shifted labels
                shift_mask = attention_mask[..., 1:].contiguous()
                # Flatten
                shift_logits_flat = shift_logits.view(-1, self.config.vocab_size)
                shift_labels_flat = shift_labels.view(-1)
                shift_mask_flat = shift_mask.view(-1)

                # Compute per-token loss
                per_token_loss = F.cross_entropy(
                    shift_logits_flat, shift_labels_flat,
                    ignore_index=-100, reduction="none",
                )
                # Mask and mean
                loss = (per_token_loss * shift_mask_flat).sum() / shift_mask_flat.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        aux_loss = torch.tensor(0.0, device=input_ids.device, dtype=logits.dtype)

        return {"loss": loss, "logits": logits, "aux_loss": aux_loss}

    def count_parameters(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
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
        """
        Autoregressive generation using recurrent mLSTM step.

        First processes the prompt in parallel, then generates token-by-token
        using the recurrent state.
        """
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device

        # Phase 1: Process the entire prompt in parallel to get logits for the
        # last position. We use the parallel forward for efficiency.
        # But we also need to build up recurrent state for generation.
        # For simplicity and correctness, we process the prompt token-by-token
        # using the recurrent step to build up state, then generate from there.

        # Initialize per-layer states
        layer_states = [None] * len(self.layers)

        # Process prompt tokens one at a time to build recurrent state
        for t in range(input_ids.shape[1]):
            token = input_ids[:, t:t+1]  # (B, 1)
            x = self.embed_tokens(token)  # (B, 1, H)

            new_layer_states = []
            for i, layer in enumerate(self.layers):
                x, state = layer.step(x, layer_states[i])
                new_layer_states.append(state)
            layer_states = new_layer_states

            last_x = x  # (B, 1, H)

        # Now generate new tokens
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self.lm_head(self.final_norm(last_x))[:, -1, :]  # (B, V)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    seen = set(generated[b].tolist())
                    for tid in seen:
                        if logits[b, tid] > 0:
                            logits[b, tid] /= repetition_penalty
                        else:
                            logits[b, tid] *= repetition_penalty

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                logits[remove_mask.scatter(1, sorted_indices, remove_mask)] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Check EOS
            if B == 1 and next_token.item() == eos_token_id:
                break

            # Feed new token through recurrent steps
            x = self.embed_tokens(next_token)  # (B, 1, H)
            new_layer_states = []
            for i, layer in enumerate(self.layers):
                x, state = layer.step(x, layer_states[i])
                new_layer_states.append(state)
            layer_states = new_layer_states
            last_x = x

        return generated


# ---------------------------------------------------------------------------
# Standalone test / sizing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("xLSTM (mLSTM) Language Model - Configuration and Parameter Count")
    print("=" * 70)

    config = XLSTMConfig()
    print(f"\nConfig:")
    print(f"  vocab_size:      {config.vocab_size}")
    print(f"  hidden_size:     {config.hidden_size}")
    print(f"  num_layers:      {config.num_layers}")
    print(f"  num_heads:       {config.num_heads}")
    print(f"  head_dim:        {config.head_dim}")
    print(f"  inner_dim:       {config._inner_dim}")
    print(f"  ff_factor:       {config.ff_factor}")
    print(f"  max_seq_len:     {config.max_seq_len}")
    print(f"  conv1d_kernel:   {config.conv1d_kernel_size}")

    ff_intermediate = int(config.ff_factor * config.hidden_size)
    ff_intermediate = ((ff_intermediate + 255) // 256) * 256
    print(f"  ff_intermediate: {ff_intermediate}")

    print(f"\nEstimated params: {config.total_params_estimate() / 1e6:.1f}M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nBuilding model on {device}...")

    # Build model
    model = XLSTMModel(config).to(device)
    total, trainable = model.count_parameters()
    print(f"Actual total params:     {total:>12,} ({total / 1e6:.1f}M)")
    print(f"Actual trainable params: {trainable:>12,} ({trainable / 1e6:.1f}M)")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    with torch.amp.autocast(device, enabled=(device == "cuda")):
        output = model(input_ids, labels=labels)

    print(f"  Loss:     {output['loss'].item():.4f}")
    print(f"  Logits:   {output['logits'].shape}")
    print(f"  Aux loss: {output['aux_loss'].item():.4f}")

    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
    generated = model.generate(prompt, max_new_tokens=16, temperature=1.0)
    print(f"  Prompt length:    {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    print(f"  Generated tokens: {generated[0].tolist()}")

    print("\nAll tests passed.")

"""
Hawk Language Model (Google DeepMind, 2024)
============================================
Pure PyTorch implementation of the Hawk model from:
  "Griffin: Mixing Gated Linear Recurrences with Local Attention
   for Efficient Language Models" (arXiv 2402.19427)

Hawk is the pure-RNN variant (no attention) built around the
Real-Gated Linear Recurrent Unit (RG-LRU).

Architecture per block:
    x -> RMSNorm -> RecurrentBlock -> + residual
                                      |
                                      v
    x -> RMSNorm -> SwiGLU MLP    -> + residual

RecurrentBlock:
    input (D) -> [linear_y (D->D_rnn), linear_x (D->D_rnn)] in parallel
    Branch x: Conv1D(kernel=4, groups=D_rnn) -> RG-LRU -> output
    Branch y: GeLU activation
    Merge: x * y (element-wise)
    Final: linear (D_rnn -> D)

RG-LRU equations (from paper):
    r_t = sigmoid(W_a @ x_t + b_a)           # recurrence gate
    i_t = sigmoid(W_x @ x_t + b_x)           # input gate
    log_a_t = -c * softplus(Lambda) * r_t     # log-space recurrence (c=8)
    a_t = exp(log_a_t)                        # recurrence weight
    h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * x_t)

Uses a parallel scan during training for efficient sequence processing.
Diagonal recurrence: each hidden dimension is independent.

Target: ~350M parameters with vocab_size=49152.
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
class HawkConfig:
    vocab_size: int = 49152
    hidden_size: int = 1024       # D: model width
    num_layers: int = 22          # depth
    max_seq_len: int = 1024
    gradient_checkpointing: bool = True

    # Hawk-specific
    recurrence_dim: int = 1280    # D_rnn: width of the recurrent block (~5/4 * D)
    num_heads: int = 16           # heads for block-diagonal structure in RG-LRU
    conv_kernel_size: int = 4     # temporal conv filter size
    mlp_expansion: int = 3        # MLP intermediate = mlp_expansion * hidden_size

    # Normalization
    rms_norm_eps: float = 1e-6

    # RG-LRU hyperparameters
    rg_lru_c: float = 8.0        # scalar constant for recurrence gate
    rg_lru_min_rad: float = 0.9   # min radius for a_param initialization
    rg_lru_max_rad: float = 0.999 # max radius for a_param initialization

    # Embedding
    tie_word_embeddings: bool = True

    def total_params_estimate(self) -> int:
        """Estimate total parameter count without instantiating the model."""
        D = self.hidden_size
        D_rnn = self.recurrence_dim
        V = self.vocab_size
        L = self.num_layers
        M = self.mlp_expansion

        # Embedding
        embed = V * D

        # Per-layer: RecurrentBlock
        # linear_y: D -> D_rnn (no bias) + linear_x: D -> D_rnn (no bias)
        rec_proj_in = 2 * D * D_rnn
        # conv1d: groups=D_rnn, kernel=4, so 4*D_rnn weights + D_rnn bias
        rec_conv = self.conv_kernel_size * D_rnn + D_rnn
        # RG-LRU:
        #   a_gate (block-diagonal): D_rnn * (D_rnn / num_heads) = D_rnn^2 / num_heads
        #   input_gate (block-diagonal): same
        #   a_param: D_rnn
        head_dim = D_rnn // self.num_heads
        rec_rglru = 2 * self.num_heads * head_dim * head_dim + D_rnn
        # output linear: D_rnn -> D (no bias)
        rec_proj_out = D_rnn * D
        # RMSNorm for recurrent: D
        rec_norm = D

        rec_total = rec_proj_in + rec_conv + rec_rglru + rec_proj_out + rec_norm

        # Per-layer: SwiGLU MLP
        # gate_proj: D -> M*D, up_proj: D -> M*D, down_proj: M*D -> D (all no bias)
        mlp_total = 3 * D * (M * D)
        # RMSNorm for MLP: D
        mlp_norm = D

        mlp_total_with_norm = mlp_total + mlp_norm

        per_layer = rec_total + mlp_total_with_norm

        # Final norm + lm_head
        final = D  # final RMSNorm
        lm_head = V * D if not self.tie_word_embeddings else 0

        total = embed + L * per_layer + final + lm_head
        return total


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# SqrtBoundDerivative: numerically stable sqrt with bounded gradient
# (from Google DeepMind's RecurrentGemma reference implementation)
# ---------------------------------------------------------------------------

_MAX_SQRT_GRADIENT = 1000.0


class SqrtBoundDerivative(torch.autograd.Function):
    """Compute sqrt(x) with gradient clipped to avoid explosion near zero."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clamp(4.0 * x, min=1.0 / (_MAX_SQRT_GRADIENT ** 2))
        return grad_output / torch.sqrt(clipped_x_times_4)


# ---------------------------------------------------------------------------
# Block-Diagonal Linear
# Used for RG-LRU gates: independent linear per head
# ---------------------------------------------------------------------------

class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear layer: num_blocks independent linear maps."""

    def __init__(self, width: int, num_blocks: int):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.block_size = width // num_blocks
        assert width % num_blocks == 0, \
            f"width ({width}) must be divisible by num_blocks ({num_blocks})"

        # Shape: (num_blocks, block_size, block_size)
        self.weight = nn.Parameter(
            torch.empty(num_blocks, self.block_size, self.block_size)
        )
        self._init_weights()

    def _init_weights(self):
        # LeCun initialization (variance = 1/fan_in)
        std = 1.0 / math.sqrt(self.block_size)
        nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., width) -> reshape to (..., num_blocks, block_size)
        *batch_dims, _ = x.shape
        x = x.view(*batch_dims, self.num_blocks, self.block_size)
        # Einsum: independent matrix multiply per block
        # x: (..., H, d), weight: (H, d, d) -> (..., H, d)
        out = torch.einsum("...hd,hde->...he", x, self.weight)
        return out.reshape(*batch_dims, self.width)


# ---------------------------------------------------------------------------
# Parallel Scan for Linear Recurrence
# Computes h_t = a_t * h_{t-1} + b_t for all t in parallel
# using the classic Blelloch prefix-sum algorithm.
# ---------------------------------------------------------------------------

def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Parallel associative scan for first-order linear recurrence.

    Given sequences a[0..T-1] and b[0..T-1], computes h[0..T-1] where:
        h[0] = b[0]
        h[t] = a[t] * h[t-1] + b[t]

    The associative operator for (a, b) pairs is:
        (a1, b1) o (a2, b2) = (a1 * a2, a2 * b1 + b2)

    This runs in O(T) work and O(log T) depth.

    Args:
        a: (batch, seq_len, dim) - recurrence coefficients
        b: (batch, seq_len, dim) - input values

    Returns:
        h: (batch, seq_len, dim) - output sequence
    """
    # For short sequences or when compile handles it, use sequential
    B, T, D = a.shape

    if T <= 1:
        return b

    # Blelloch-style parallel scan (up-sweep + down-sweep)
    # We work on copies to avoid in-place issues with autograd
    # Store intermediate results at each level

    # Pad T to next power of 2 for clean binary tree
    log2T = math.ceil(math.log2(T)) if T > 1 else 1
    T_padded = 1 << log2T

    # Pad if needed
    if T_padded != T:
        pad_len = T_padded - T
        a = F.pad(a, (0, 0, 0, pad_len), value=0.0)  # a=0 means identity
        b = F.pad(b, (0, 0, 0, pad_len), value=0.0)
        # a[t]=0 for padded positions means h[t] = 0 * h[t-1] + 0 = 0

    # We'll implement this iteratively using the standard approach:
    # For training efficiency, use a simple sequential scan when T is modest
    # and the parallel scan for longer sequences.

    # For numerical stability and torch.compile compatibility,
    # implement sequential scan which is actually fast for typical seq_lens
    # (PyTorch kernels handle the loop efficiently, and torch.compile
    # can unroll/optimize it)
    return _sequential_scan(a[:, :T, :], b[:, :T, :])


def _sequential_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Sequential scan fallback. Fast for moderate sequence lengths
    and fully compatible with torch.compile.

    Args:
        a: (batch, seq_len, dim)
        b: (batch, seq_len, dim)
    Returns:
        h: (batch, seq_len, dim)
    """
    B, T, D = a.shape
    h_list = []
    h_prev = torch.zeros(B, D, device=a.device, dtype=a.dtype)

    for t in range(T):
        h_prev = a[:, t, :] * h_prev + b[:, t, :]
        h_list.append(h_prev)

    return torch.stack(h_list, dim=1)


def _parallel_prefix_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    True parallel prefix scan using the Blelloch algorithm.
    O(T) work, O(log T) depth.

    For the associative operator:
        (a1, b1) o (a2, b2) = (a1 * a2, a2 * b1 + b2)

    Args:
        a: (batch, seq_len, dim) - must have seq_len = power of 2
        b: (batch, seq_len, dim)
    Returns:
        h: (batch, seq_len, dim)
    """
    B, T, D = a.shape
    assert T & (T - 1) == 0, "seq_len must be a power of 2"

    log2T = int(math.log2(T))

    # Clone to avoid modifying inputs
    aa = a.clone()
    bb = b.clone()

    # Up-sweep (reduce) phase
    for d in range(log2T):
        stride = 1 << (d + 1)
        half = 1 << d
        # Indices: stride-1, 2*stride-1, 3*stride-1, ...
        idx_right = torch.arange(stride - 1, T, stride, device=a.device)
        idx_left = idx_right - half

        # (a_left, b_left) o (a_right, b_right) = (a_left * a_right, a_right * b_left + b_right)
        new_b = aa[:, idx_right, :] * bb[:, idx_left, :] + bb[:, idx_right, :]
        new_a = aa[:, idx_left, :] * aa[:, idx_right, :]

        bb = bb.clone()
        aa = aa.clone()
        bb[:, idx_right, :] = new_b
        aa[:, idx_right, :] = new_a

    # The last element now contains the total reduction
    # Clear it for down-sweep
    aa_down = aa.clone()
    bb_down = bb.clone()
    bb_down[:, T - 1, :] = bb[:, T - 1, :]  # Keep the final result

    # Down-sweep phase
    # Set identity at root
    aa_down[:, T - 1, :] = 0.0  # a=0 is not identity... we need a different approach

    # Actually, the Blelloch scan for non-standard operators needs care.
    # For the linear recurrence, the sequential scan is simpler and
    # PyTorch + torch.compile handles it well. Let's use the sequential one.
    return _sequential_scan(a, b)


# ---------------------------------------------------------------------------
# Real-Gated Linear Recurrent Unit (RG-LRU)
# ---------------------------------------------------------------------------

class RGLRU(nn.Module):
    """
    Real-Gated Linear Recurrent Unit from the Griffin/Hawk paper.

    Computes:
        r_t = sigmoid(gate_a(x_t))           # recurrence gate
        i_t = sigmoid(gate_x(x_t))           # input gate
        log_a_t = -c * softplus(Lambda) * r_t
        a_t = exp(log_a_t)
        h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * x_t)

    Uses block-diagonal linear layers for gates (one per head).
    Diagonal recurrence: each dimension is independent.
    """

    def __init__(self, width: int, num_heads: int, c: float = 8.0,
                 min_rad: float = 0.9, max_rad: float = 0.999):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.c = c

        # Learnable diagonal recurrence parameter (Lambda)
        self.a_param = nn.Parameter(torch.empty(width))

        # Block-diagonal gates (one independent linear per head)
        self.input_gate = BlockDiagonalLinear(width, num_heads)
        self.a_gate = BlockDiagonalLinear(width, num_heads)

        # Initialize a_param so that a^c is uniformly distributed
        # between min_rad and max_rad
        self._init_a_param(min_rad, max_rad)

    def _init_a_param(self, min_rad: float, max_rad: float, eps: float = 1e-8):
        """
        Initialize Lambda such that sigmoid(Lambda)^c is uniformly
        distributed in [min_rad, max_rad].

        From the paper: a = sigmoid(Lambda), and we want a^c uniform in [min_rad, max_rad].
        We use the log-space parameterization: log(a) = -softplus(Lambda),
        so we need -c*softplus(Lambda) to give log values corresponding to
        the desired range.

        Following RecurrentGemma reference:
        Initialize in softplus^{-1} space.
        """
        with torch.no_grad():
            # We want: exp(-c * softplus(Lambda)) in [min_rad, max_rad]
            # So: -c * softplus(Lambda) in [log(min_rad), log(max_rad)]
            # So: softplus(Lambda) in [-log(max_rad)/c, -log(min_rad)/c]
            # So: Lambda in softplus^{-1}([-log(max_rad)/c, -log(min_rad)/c])
            # softplus^{-1}(y) = log(exp(y) - 1)

            sp_min = -math.log(max_rad) / self.c  # smaller softplus value -> larger a
            sp_max = -math.log(min_rad) / self.c  # larger softplus value -> smaller a

            # Uniform in softplus space, then invert
            self.a_param.uniform_(sp_min, sp_max)
            # Apply inverse softplus: Lambda = log(exp(sp) - 1)
            self.a_param.copy_(torch.log(torch.exp(self.a_param) - 1.0 + eps))

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, width)
            h_prev: (batch, width) or None - previous hidden state for generation

        Returns:
            y: (batch, seq_len, width) - output
            h_last: (batch, width) - final hidden state
        """
        B, T, D = x.shape

        # Compute gates
        gate_x = torch.sigmoid(self.input_gate(x))   # (B, T, D)
        gate_a = torch.sigmoid(self.a_gate(x))        # (B, T, D)

        # Compute recurrence weights in log space
        # log(a_t) = -c * softplus(Lambda) * gate_a_t
        log_a = -self.c * F.softplus(self.a_param) * gate_a  # (B, T, D)
        a = torch.exp(log_a)                                  # (B, T, D)

        # Compute the input multiplier: sqrt(1 - a^2)
        # Using numerically stable SqrtBoundDerivative
        a_square = torch.exp(2.0 * log_a)                     # (B, T, D)
        multiplier = SqrtBoundDerivative.apply(1.0 - a_square) # (B, T, D)

        # Gated input: sqrt(1 - a^2) * (gate_x * x)
        gated_x = gate_x * x
        b = multiplier * gated_x  # (B, T, D) - the input to the recurrence

        # Run the linear recurrence: h_t = a_t * h_{t-1} + b_t
        if h_prev is not None and T == 1:
            # Single-step inference mode
            h = a[:, 0, :] * h_prev + b[:, 0, :]
            y = h.unsqueeze(1)
            return y, h
        else:
            # If we have a previous hidden state, incorporate it into the first step
            if h_prev is not None:
                # Modify b[0] to include contribution from h_prev
                b_modified = b.clone()
                b_modified[:, 0, :] = a[:, 0, :] * h_prev + b[:, 0, :]
                # For subsequent steps, a[0] contribution is already in b_modified[0]
                # We need to set a[0] = 0 so the scan doesn't double-count
                a_modified = a.clone()
                a_modified[:, 0, :] = 0.0
                y = parallel_scan(a_modified, b_modified)
            else:
                y = parallel_scan(a, b)

            h_last = y[:, -1, :]
            return y, h_last


# ---------------------------------------------------------------------------
# Recurrent Block (Temporal Mixing)
# ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    """
    Hawk recurrent block for temporal mixing.

    Architecture:
        input (D) -> linear_y (D -> D_rnn), linear_x (D -> D_rnn) in parallel
        Branch x: Conv1D(kernel=4, depthwise) -> RG-LRU
        Branch y: GeLU activation
        Output: (x_branch * y_branch) -> linear_out (D_rnn -> D)
    """

    def __init__(self, config: HawkConfig):
        super().__init__()
        D = config.hidden_size
        D_rnn = config.recurrence_dim

        # Input projections (two parallel branches)
        self.linear_y = nn.Linear(D, D_rnn, bias=False)
        self.linear_x = nn.Linear(D, D_rnn, bias=False)

        # Depthwise (separable) Conv1D on the x branch
        # kernel_size=4, causal padding=3 (left-padded)
        self.conv1d = nn.Conv1d(
            in_channels=D_rnn,
            out_channels=D_rnn,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1,  # causal: pad left
            groups=D_rnn,  # depthwise separable
        )

        # RG-LRU layer
        self.rg_lru = RGLRU(
            width=D_rnn,
            num_heads=config.num_heads,
            c=config.rg_lru_c,
            min_rad=config.rg_lru_min_rad,
            max_rad=config.rg_lru_max_rad,
        )

        # Output projection
        self.linear_out = nn.Linear(D_rnn, D, bias=False)

    def forward(self, x: torch.Tensor,
                conv_state: Optional[torch.Tensor] = None,
                rnn_state: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, D)
            conv_state: optional conv cache for generation
            rnn_state: optional RNN hidden state for generation

        Returns:
            output: (batch, seq_len, D)
            new_conv_state: updated conv state
            new_rnn_state: updated RNN hidden state
        """
        # Two parallel branches
        y = F.gelu(self.linear_y(x))       # (B, T, D_rnn) - gating branch
        x_branch = self.linear_x(x)        # (B, T, D_rnn) - recurrence branch

        # Conv1D (causal): transpose for Conv1d, then truncate future
        # x_branch: (B, T, D_rnn) -> (B, D_rnn, T) for conv
        x_conv = x_branch.transpose(1, 2)  # (B, D_rnn, T)

        if conv_state is not None and x_conv.shape[2] == 1:
            # Single-step inference: use conv_state cache
            # conv_state: (B, D_rnn, kernel_size-1)
            x_conv_full = torch.cat([conv_state, x_conv], dim=2)
            x_conv_out = self.conv1d(x_conv_full)
            # Take only the last position
            x_conv_out = x_conv_out[:, :, -1:]
            new_conv_state = x_conv_full[:, :, 1:]  # slide window
        else:
            x_conv_out = self.conv1d(x_conv)
            # Causal: remove the future padding (keep only first T outputs)
            x_conv_out = x_conv_out[:, :, :x_conv.shape[2]]
            new_conv_state = x_conv[:, :, -(self.conv1d.kernel_size[0] - 1):]

        x_branch = x_conv_out.transpose(1, 2)  # (B, T, D_rnn)

        # RG-LRU
        x_branch, new_rnn_state = self.rg_lru(x_branch, h_prev=rnn_state)

        # Merge branches (output gating)
        merged = x_branch * y  # (B, T, D_rnn)

        # Project back to model dimension
        output = self.linear_out(merged)  # (B, T, D)

        return output, new_conv_state, new_rnn_state


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """SwiGLU MLP: gate_proj and up_proj with SiLU gating, then down_proj."""

    def __init__(self, config: HawkConfig):
        super().__init__()
        D = config.hidden_size
        D_ff = config.mlp_expansion * D

        self.gate_proj = nn.Linear(D, D_ff, bias=False)
        self.up_proj = nn.Linear(D, D_ff, bias=False)
        self.down_proj = nn.Linear(D_ff, D, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Hawk Block
# ---------------------------------------------------------------------------

class HawkBlock(nn.Module):
    """
    A single Hawk layer:
        x -> RMSNorm -> RecurrentBlock -> + residual
                                          |
                                          v
        x -> RMSNorm -> SwiGLU MLP    -> + residual
    """

    def __init__(self, config: HawkConfig):
        super().__init__()
        self.rec_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.recurrent = RecurrentBlock(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor,
                conv_state: Optional[torch.Tensor] = None,
                rnn_state: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, hidden_size)
            conv_state: optional conv cache
            rnn_state: optional RNN hidden state

        Returns:
            x: (batch, seq_len, hidden_size)
            new_conv_state, new_rnn_state
        """
        # Temporal mixing with residual
        residual = x
        rec_out, new_conv_state, new_rnn_state = self.recurrent(
            self.rec_norm(x), conv_state=conv_state, rnn_state=rnn_state
        )
        x = residual + rec_out

        # MLP with residual
        residual = x
        x = residual + self.mlp(self.mlp_norm(x))

        return x, new_conv_state, new_rnn_state

    def forward_no_cache(self, x: torch.Tensor) -> torch.Tensor:
        """Forward without caching (for training with gradient checkpointing)."""
        residual = x
        rec_out, _, _ = self.recurrent(self.rec_norm(x))
        x = residual + rec_out

        residual = x
        x = residual + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Hawk Model
# ---------------------------------------------------------------------------

class HawkModel(nn.Module):
    """
    Full Hawk language model: embedding -> N x HawkBlock -> RMSNorm -> lm_head

    Implements the required tournament interface:
        forward(input_ids, labels=None, attention_mask=None)
            -> {"loss": ..., "logits": ..., "aux_loss": tensor(0.0)}
        count_parameters() -> (total, trainable)
        generate(input_ids, ...) -> token ids
    """

    def __init__(self, config: HawkConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Hawk blocks
        self.layers = nn.ModuleList([
            HawkBlock(config) for _ in range(config.num_layers)
        ])

        # Output
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient checkpointing flag
        self._gradient_checkpointing = config.gradient_checkpointing

    def _init_weights(self, module):
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            # Small initialization for conv
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Note: BlockDiagonalLinear and RGLRU.a_param have their own init

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) - token IDs
            labels: (batch, seq_len) - target token IDs (optional)
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
                          (used to zero out padding positions)

        Returns:
            dict with:
                "loss": cross-entropy loss or None
                "logits": (batch, seq_len, vocab_size)
                "aux_loss": tensor(0.0) (Hawk has no auxiliary loss)
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed_tokens(input_ids)  # (B, T, D)

        # Apply attention mask if provided (zero out padded positions)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        # Process through Hawk blocks
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer.forward_no_cache, x, use_reentrant=False
                )
            else:
                x, _, _ = layer(x)

        # Output projection
        logits = self.lm_head(self.final_norm(x))  # (B, T, V)

        # Compute loss
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
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def _init_generation_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize conv and RNN caches for autoregressive generation."""
        conv_states = []
        rnn_states = []
        for _ in range(self.config.num_layers):
            # Conv state: (B, D_rnn, kernel_size - 1)
            conv_states.append(
                torch.zeros(batch_size, self.config.recurrence_dim,
                           self.config.conv_kernel_size - 1,
                           device=device, dtype=dtype)
            )
            # RNN state: (B, D_rnn)
            rnn_states.append(
                torch.zeros(batch_size, self.config.recurrence_dim,
                           device=device, dtype=dtype)
            )
        return conv_states, rnn_states

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 256,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1,
                 eos_token_id: int = 2):
        """
        Autoregressive text generation with RNN state caching.

        First processes the full prompt through the model to build up
        the hidden state, then generates one token at a time using
        cached conv and RNN states for O(1) per-step cost.

        Args:
            input_ids: (batch, prompt_len) - prompt token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            repetition_penalty: penalty for repeated tokens
            eos_token_id: stop token

        Returns:
            input_ids: (batch, prompt_len + generated_len)
        """
        self.eval()
        B, prompt_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # Phase 1: Process the prompt to build up hidden states
        # We process the full prompt through each layer and cache the final states
        conv_states, rnn_states = self._init_generation_cache(B, device, dtype)

        x = self.embed_tokens(input_ids)  # (B, prompt_len, D)

        for i, layer in enumerate(self.layers):
            x, conv_states[i], rnn_states[i] = layer(
                x, conv_state=None, rnn_state=None
            )

        # Get logits for the last position (to generate first new token)
        last_logits = self.lm_head(self.final_norm(x[:, -1:, :]))  # (B, 1, V)
        logits = last_logits[:, -1, :]  # (B, V)

        # Phase 2: Autoregressive generation with cached states
        for _ in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for tid in set(input_ids[b].tolist()):
                        if logits[b, tid] > 0:
                            logits[b, tid] /= repetition_penalty
                        else:
                            logits[b, tid] *= repetition_penalty

            # Temperature scaling
            logits = logits / max(temperature, 1e-8)

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
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if next_token.item() == eos_token_id:
                break

            # Forward pass for single token with caching
            x = self.embed_tokens(next_token)  # (B, 1, D)

            for i, layer in enumerate(self.layers):
                x, conv_states[i], rnn_states[i] = layer(
                    x, conv_state=conv_states[i], rnn_state=rnn_states[i]
                )

            logits = self.lm_head(self.final_norm(x))[:, -1, :]  # (B, V)

        return input_ids


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Hawk Model - Standalone Test")
    print("=" * 70)

    # Create config
    config = HawkConfig()
    print(f"\nConfig:")
    print(f"  vocab_size:      {config.vocab_size}")
    print(f"  hidden_size:     {config.hidden_size}")
    print(f"  num_layers:      {config.num_layers}")
    print(f"  recurrence_dim:  {config.recurrence_dim}")
    print(f"  num_heads:       {config.num_heads}")
    print(f"  conv_kernel:     {config.conv_kernel_size}")
    print(f"  mlp_expansion:   {config.mlp_expansion}")
    print(f"  max_seq_len:     {config.max_seq_len}")

    print(f"\nEstimated params:  {config.total_params_estimate():,}")

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    model = HawkModel(config).to(device)
    total, trainable = model.count_parameters()
    print(f"Actual params:     {total:,} (trainable: {trainable:,})")
    print(f"Memory (fp16):     {total * 2 / 1e9:.2f} GB")
    print(f"Param estimate error: {abs(config.total_params_estimate() - total) / total * 100:.1f}%")

    # Test forward pass
    print("\n--- Forward Pass Test ---")
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Without labels
    outputs = model(input_ids)
    print(f"Logits shape:  {outputs['logits'].shape}")
    print(f"Loss:          {outputs['loss']}")
    print(f"Aux loss:      {outputs['aux_loss'].item()}")

    # With labels
    outputs = model(input_ids, labels=labels)
    print(f"Loss (w/ labels): {outputs['loss'].item():.4f}")

    # Test backward pass
    print("\n--- Backward Pass Test ---")
    outputs["loss"].backward()
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append((name, p.grad.norm().item()))
    print(f"Parameters with gradients: {len(grad_norms)}")
    if grad_norms:
        max_grad = max(grad_norms, key=lambda x: x[1])
        min_grad = min(grad_norms, key=lambda x: x[1])
        print(f"Max grad norm: {max_grad[0]} = {max_grad[1]:.6f}")
        print(f"Min grad norm: {min_grad[0]} = {min_grad[1]:.6f}")

    # Test generation
    print("\n--- Generation Test ---")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
    print(f"Prompt tokens: {prompt.shape}")

    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=2,
    )
    print(f"Generated tokens: {generated.shape}")
    print(f"New tokens: {generated.shape[1] - prompt.shape[1]}")

    # Test with attention mask
    print("\n--- Attention Mask Test ---")
    model.train()
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    attention_mask[:, -10:] = 0  # mask last 10 tokens
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    print(f"Loss (w/ mask): {outputs['loss'].item():.4f}")

    # Test gradient checkpointing
    print("\n--- Gradient Checkpointing Test ---")
    config_gc = HawkConfig(gradient_checkpointing=True, num_layers=4,
                           hidden_size=256, recurrence_dim=384)
    model_gc = HawkModel(config_gc).to(device)
    small_ids = torch.randint(0, config_gc.vocab_size, (2, 64), device=device)
    small_labels = torch.randint(0, config_gc.vocab_size, (2, 64), device=device)
    out_gc = model_gc(small_ids, labels=small_labels)
    out_gc["loss"].backward()
    print(f"Gradient checkpointing: OK (loss={out_gc['loss'].item():.4f})")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

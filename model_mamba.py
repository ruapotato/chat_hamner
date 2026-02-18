"""
Mamba-2 (Selective State Space Model) Language Model
=====================================================
Pure PyTorch implementation of Mamba-2 based on "Transformers are SSMs:
Generalized Models and Efficient Algorithms Through Structured State Space
Duality" (Tri Dao, Albert Gu, 2024).

Key architectural points:
- Structured State Space Duality (SSD): the core SSM can be computed as
  chunked matrix multiplications for efficient parallel training.
- Multi-head SSM with scalar-times-identity A matrix per head.
- Input-dependent B, C, and dt (delta/timestep) parameters.
- Depthwise 1-D convolution on the (x, B, C) path before the SSM.
- Gated output via SiLU(z) * norm(y), same as the reference implementation.
- No attention layers whatsoever -- pure SSM with O(n) complexity.

References:
  [1] https://arxiv.org/abs/2405.21060
  [2] https://github.com/tommyip/mamba2-minimal (Tommy Ip)
  [3] https://github.com/state-spaces/mamba (official repo)
  [4] https://github.com/johnma2006/mamba-minimal (John Ma)
  [5] https://tridao.me/blog/2024/mamba2-part1-model/
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
class MambaConfig:
    vocab_size: int = 49152
    hidden_size: int = 1152       # D -- model dimension
    num_layers: int = 36          # number of Mamba-2 blocks
    max_seq_len: int = 1024
    gradient_checkpointing: bool = True

    # Mamba-2 specific
    d_state: int = 64             # N -- SSM state dimension (Mamba-2 uses 64-256)
    d_conv: int = 4               # convolution kernel width
    expand: int = 2               # expansion factor  (d_inner = expand * hidden_size)
    headdim: int = 64             # P -- head dimension
    chunk_size: int = 64          # Q -- chunk length for SSD algorithm

    # Regularization / init
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    conv_bias: bool = True

    def __post_init__(self):
        self.d_inner = self.expand * self.hidden_size
        assert self.d_inner % self.headdim == 0, (
            f"d_inner ({self.d_inner}) must be divisible by headdim ({self.headdim})"
        )
        self.nheads = self.d_inner // self.headdim

    def total_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        d = self.hidden_size
        di = self.d_inner
        ns = self.d_state
        nh = self.nheads
        d_in_proj = 2 * di + 2 * ns + nh
        conv_dim = di + 2 * ns

        per_layer = (
            d * d_in_proj              # in_proj
            + conv_dim * self.d_conv   # conv weight
            + conv_dim                 # conv bias
            + nh * 3                   # dt_bias + A_log + D
            + di                       # inner norm weight
            + di * d                   # out_proj
            + d                        # layer norm weight
        )
        embed = self.vocab_size * d
        total = embed + per_layer * self.num_layers + d  # +d for final norm
        if not self.tie_word_embeddings:
            total += embed
        return total


# ---------------------------------------------------------------------------
# RMSNorm  (optionally gated, matching Mamba-2 reference)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization, with optional gating."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        if gate is not None:
            x = x * F.silu(gate)
        # Compute in float32 for numerical stability
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# SSD Algorithm  (Structured State Space Duality -- the core of Mamba-2)
# ---------------------------------------------------------------------------

def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum calculation.

    exp(segsum(A)) produces a 1-semiseparable matrix equivalent to a scalar SSM.
    See https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py

    Args:
        x: (..., T)
    Returns:
        (..., T, T) lower-triangular segment-sum matrix
    """
    T = x.size(-1)
    # Replicate along a new trailing dimension
    x = x.unsqueeze(-1).expand(*x.shape, T)           # (..., T, T)
    # Zero out upper triangle (keep strict lower triangle)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0.0)
    # Cumulative sum along the second-to-last dim
    x_segsum = torch.cumsum(x, dim=-2)
    # Mask upper triangle with -inf so that exp(...) gives 0
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(
    x: torch.Tensor,       # (batch, seqlen, nheads, headdim)
    A: torch.Tensor,        # (batch, seqlen, nheads)
    B: torch.Tensor,        # (batch, seqlen, nheads, d_state)
    C: torch.Tensor,        # (batch, seqlen, nheads, d_state)
    chunk_size: int,
    initial_states: Optional[torch.Tensor] = None,
) -> tuple:
    """Structured State Space Duality -- the parallel SSD algorithm.

    Processes the sequence in chunks of size `chunk_size`, computing intra-chunk
    outputs via matrix multiplication and chaining inter-chunk states via a
    recurrence over chunk boundaries.

    Returns:
        y:           (batch, seqlen, nheads, headdim)
        final_state: (batch, nheads, headdim, d_state)
    """
    batch, seqlen, nheads, headdim = x.shape
    assert seqlen % chunk_size == 0, (
        f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
    )

    # Reshape into chunks:  (batch, n_chunks, chunk_size, ...)
    x, A, B, C = [
        t.reshape(batch, seqlen // chunk_size, chunk_size, *t.shape[2:])
        for t in (x, A, B, C)
    ]
    # A: (batch, n_chunks, chunk_size, nheads)  ->  (batch, nheads, n_chunks, chunk_size)
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # ------------------------------------------------------------------
    # 1. Intra-chunk (diagonal blocks): the SSD within each chunk
    # ------------------------------------------------------------------
    # L is the lower-triangular causal mask weighted by decayed A values
    L = torch.exp(segsum(A))  # (batch, nheads, n_chunks, chunk_size, chunk_size)
    # Y_diag = C^T @ (L .* (B @ X))  -- the "attention-like" formulation
    Y_diag = torch.einsum(
        "bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x
    )

    # ------------------------------------------------------------------
    # 2. Inter-chunk state accumulation (B terms)
    # ------------------------------------------------------------------
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    # states: (batch, n_chunks, nheads, headdim, d_state)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # ------------------------------------------------------------------
    # 3. Inter-chunk SSM recurrence (A terms)
    # ------------------------------------------------------------------
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)

    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # ------------------------------------------------------------------
    # 4. State-to-output conversion per chunk (C terms)
    # ------------------------------------------------------------------
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Combine intra-chunk and inter-chunk contributions
    Y = Y_diag + Y_off
    Y = Y.reshape(batch, seqlen, nheads, headdim)

    return Y, final_state


# ---------------------------------------------------------------------------
# Mamba-2 Layer
# ---------------------------------------------------------------------------

class Mamba2Layer(nn.Module):
    """A single Mamba-2 selective SSM layer.

    Architecture (per the SSD paper, Section 8):
      1. Linear projection producing (z, x, B, C, dt) in parallel.
      2. Depthwise conv1d on (x, B, C).
      3. SiLU activation on (x, B, C).
      4. SSD core: x * dt passed through the SSM with parameters A, B, C.
      5. Skip connection: y += D * x.
      6. Gated normalization: output = out_proj(norm(y, gate=z)).
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        di = config.d_inner
        ns = config.d_state
        nh = config.nheads

        # Single projection: z (gate), x (input), B, C, dt all at once
        # Dimensions: z=d_inner, x=d_inner, B=d_state, C=d_state, dt=nheads
        d_in_proj = 2 * di + 2 * ns + nh
        self.in_proj = nn.Linear(d, d_in_proj, bias=False)

        # Depthwise convolution on (x || B || C)
        conv_dim = di + 2 * ns
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=config.d_conv,
            groups=conv_dim,           # depthwise
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )

        # Per-head learnable parameters
        self.dt_bias = nn.Parameter(torch.empty(nh))
        self.A_log = nn.Parameter(torch.empty(nh))
        self.D = nn.Parameter(torch.empty(nh))

        # Gated normalization and output projection
        self.norm = RMSNorm(di, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(di, d, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, seqlen, hidden_size)
        Returns:
            y: (batch, seqlen, hidden_size)
        """
        batch, seqlen, _ = u.shape
        config = self.config
        chunk_size = config.chunk_size

        # Pad sequence to multiple of chunk_size if needed
        pad_len = (chunk_size - seqlen % chunk_size) % chunk_size
        if pad_len > 0:
            u = F.pad(u, (0, 0, 0, pad_len))
        padded_len = u.shape[1]

        A = -torch.exp(self.A_log.float())  # (nheads,)

        # Project input -> (z, xBC, dt)
        zxbcdt = self.in_proj(u)  # (batch, padded_len, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [config.d_inner, config.d_inner + 2 * config.d_state, config.nheads],
            dim=-1,
        )

        # Causal depthwise conv1d on (x || B || C)
        xBC = xBC.transpose(1, 2)                        # (batch, conv_dim, padded_len)
        xBC = self.conv1d(xBC)[:, :, :padded_len]        # causal: trim future
        xBC = xBC.transpose(1, 2)                        # (batch, padded_len, conv_dim)
        xBC = F.silu(xBC)

        # Split into x, B, C
        x, B, C = torch.split(
            xBC,
            [config.d_inner, config.d_state, config.d_state],
            dim=-1,
        )

        # Prepare dt (softplus for positivity)
        dt = F.softplus(dt + self.dt_bias)  # (batch, padded_len, nheads)

        # Reshape x into multi-head format
        x = x.reshape(batch, padded_len, config.nheads, config.headdim)

        # Expand B, C to have a head dimension (shared across heads)
        B = B.unsqueeze(2).expand(-1, -1, config.nheads, -1)  # (batch, padded_len, nheads, d_state)
        C = C.unsqueeze(2).expand(-1, -1, config.nheads, -1)

        # Run the SSD core
        y, _ = ssd(
            x * dt.unsqueeze(-1),       # scale input by dt
            A * dt,                      # discretized A
            B,
            C,
            chunk_size=chunk_size,
        )

        # Skip connection: D * x
        y = y + x * self.D.unsqueeze(-1)  # D is per-head

        # Flatten heads back to d_inner
        y = y.reshape(batch, padded_len, config.d_inner)

        # Gated output: norm(y) * silu(z), then project
        y = self.norm(y, gate=z)
        y = self.out_proj(y)

        # Remove padding
        if pad_len > 0:
            y = y[:, :seqlen, :]

        return y


# ---------------------------------------------------------------------------
# Mamba-2 Block  (norm -> mamba_layer -> residual)
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """Pre-norm residual block wrapping a Mamba-2 layer."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = Mamba2Layer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


# ---------------------------------------------------------------------------
# Full Mamba-2 Language Model
# ---------------------------------------------------------------------------

class MambaModel(nn.Module):
    """Mamba-2 language model with the tournament-compatible interface.

    Architecture:
        token_embedding -> [Mamba2Block x num_layers] -> final_norm -> lm_head
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        config.__post_init__()  # ensure derived attributes are set

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Stack of Mamba-2 blocks
        self.layers = nn.ModuleList(
            [Mamba2Block(config) for _ in range(config.num_layers)]
        )

        # Final normalization and language model head
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Special init for A_log: log of uniform [1, nheads] following Mamba convention
        self._init_mamba_params()

        self._gradient_checkpointing = config.gradient_checkpointing

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_mamba_params(self):
        """Special initialization for Mamba-specific parameters."""
        for layer in self.layers:
            mixer = layer.mixer
            # A_log: initialize so that A = -exp(A_log) gives values in [-nheads, -1]
            # Following the Mamba-2 convention: A_log initialized to log of [1..nheads]
            with torch.no_grad():
                arange = torch.arange(1, self.config.nheads + 1, dtype=torch.float32)
                mixer.A_log.copy_(torch.log(arange))
            # dt_bias: initialize with inverse softplus of uniform [dt_min, dt_max]
            # Simplified: small positive values
            nn.init.uniform_(mixer.dt_bias, -2.0, 0.0)
            # D: ones (acts as skip connection scaling)
            nn.init.ones_(mixer.D)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids:      (batch, seqlen) token IDs
            labels:         (batch, seqlen) target IDs for cross-entropy loss
            attention_mask: (batch, seqlen) 1 for real tokens, 0 for padding
                            (used to mask loss computation only -- Mamba has no
                            attention mechanism)

        Returns:
            dict with keys: "loss", "logits", "aux_loss"
        """
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False,
                )
            else:
                x = layer(x)

        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # If attention_mask is provided, ignore padded positions
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                # Set masked labels to -100 (ignore_index)
                shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)

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
        """Autoregressive generation.

        Note: This is the simple approach -- feed the full context each step.
        For production, a recurrent (step-by-step) cache would be used. The SSD
        architecture supports O(1) per-step inference, but implementing the
        recurrent cache is not necessary for this tournament-compatible interface.

        Args:
            input_ids:        (batch, seqlen) prompt tokens
            max_new_tokens:   maximum tokens to generate
            temperature:      sampling temperature
            top_k:            top-k filtering
            top_p:            nucleus sampling threshold
            repetition_penalty: penalty for repeated tokens
            eos_token_id:     end of sequence token

        Returns:
            input_ids: (batch, seqlen + generated) full sequence
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits = self(idx_cond)["logits"][:, -1, :]  # (batch, vocab_size)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Temperature
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
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right so the first token above threshold is kept
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # Scatter back
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

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
    print("Mamba-2 Language Model -- Standalone Test")
    print("=" * 70)

    config = MambaConfig()
    print(f"\nConfig:")
    print(f"  vocab_size     = {config.vocab_size}")
    print(f"  hidden_size    = {config.hidden_size}")
    print(f"  num_layers     = {config.num_layers}")
    print(f"  d_inner        = {config.d_inner}")
    print(f"  nheads         = {config.nheads}")
    print(f"  headdim        = {config.headdim}")
    print(f"  d_state        = {config.d_state}")
    print(f"  d_conv         = {config.d_conv}")
    print(f"  chunk_size     = {config.chunk_size}")
    print(f"  max_seq_len    = {config.max_seq_len}")
    print(f"  Estimated params: {config.total_params_estimate() / 1e6:.1f}M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    model = MambaModel(config).to(device)
    total, trainable = model.count_parameters()
    print(f"Actual params:    total={total/1e6:.1f}M, trainable={trainable/1e6:.1f}M")

    # Test forward pass
    batch_size = 2
    seq_len = 128  # Must be divisible by chunk_size (64)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)

    print(f"\nForward pass: input_ids shape = {input_ids.shape}")
    output = model(input_ids, labels=labels, attention_mask=attention_mask)
    print(f"  logits shape = {output['logits'].shape}")
    print(f"  loss         = {output['loss'].item():.4f}")
    print(f"  aux_loss     = {output['aux_loss'].item():.4f}")

    # Test with non-chunk-aligned sequence length
    seq_len_odd = 100
    input_ids_odd = torch.randint(0, config.vocab_size, (batch_size, seq_len_odd), device=device)
    labels_odd = torch.randint(0, config.vocab_size, (batch_size, seq_len_odd), device=device)
    print(f"\nForward pass (non-aligned): input_ids shape = {input_ids_odd.shape}")
    output_odd = model(input_ids_odd, labels=labels_odd)
    print(f"  logits shape = {output_odd['logits'].shape}")
    print(f"  loss         = {output_odd['loss'].item():.4f}")

    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 16), device=device)
    print(f"\nGeneration: prompt shape = {prompt.shape}")
    generated = model.generate(prompt, max_new_tokens=32, temperature=1.0, top_k=50)
    print(f"  generated shape = {generated.shape}")
    print(f"  new tokens      = {generated.shape[1] - prompt.shape[1]}")

    # Test backward pass (gradient flow)
    model.train()
    model._gradient_checkpointing = False  # disable for this test
    output = model(input_ids, labels=labels)
    output["loss"].backward()
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append((name, p.grad.norm().item()))
    print(f"\nBackward pass: {len(grad_norms)} parameters received gradients")
    if grad_norms:
        print(f"  First 5 grad norms:")
        for name, norm in grad_norms[:5]:
            print(f"    {name}: {norm:.6f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

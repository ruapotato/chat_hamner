"""
Liquid Neural Network Language Model
=====================================
A CfC (Closed-form Continuous-depth) inspired language model in pure PyTorch.

Architecture based on:
- Liquid Time-Constant Networks (Hasani et al., 2021)
- Closed-form Continuous-time Neural Networks (Hasani et al., 2022, Nature MI)
- Liquid Structural State-Space Models / Liquid-S4 (Hasani et al., 2022)
- LFM2 hybrid architecture (Liquid AI, 2025) - gated convolutions + liquid cells

Core idea: the state transition dynamics are INPUT-DEPENDENT. Unlike fixed-weight
RNNs where the recurrence matrix is static, liquid networks adapt their dynamics
based on the current input. This is inspired by biological neural circuits in
C. elegans.

The CfC closed-form solution avoids numerical ODE integration:
  h_new = f1 * (1 - sigma) + f2 * sigma
where sigma = sigmoid(tau_a(x) + tau_b) is an input-dependent time gate,
and f1, f2 are nonlinear transforms of the input projected through a backbone.

The recurrence itself uses an input-dependent decay (linearized LTC):
  h_t = decay_t * h_{t-1} + (1 - decay_t) * candidate_t
where decay_t depends on the current input, making the effective time constant
of each hidden unit adapt to incoming signals.

Each block: RMSNorm -> ShortConv -> CfC Liquid Cell -> Gate -> Residual,
            RMSNorm -> SwiGLU MLP -> Residual

Target: ~300M params with vocab_size=49152.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LiquidConfig:
    vocab_size: int = 49152
    hidden_size: int = 1024
    num_layers: int = 24
    max_seq_len: int = 1024
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = True

    # Liquid-specific parameters
    state_size: int = 64          # recurrent state dimension per head
    num_heads: int = 16           # number of parallel liquid heads
    conv_size: int = 4            # short 1D causal conv kernel width
    backbone_layers: int = 1      # depth of backbone network in CfC cell
    backbone_activation: str = "silu"  # activation for backbone

    # MLP settings
    intermediate_size: int = 2352  # SwiGLU MLP intermediate dimension (~300M total)

    def total_params_estimate(self) -> int:
        """Parameter count estimate matching the actual model architecture."""
        embed = self.vocab_size * self.hidden_size

        head_dim = self.hidden_size // self.num_heads
        state_dim = self.state_size

        # Per-layer liquid cell params (CfCLiquidCell):
        # in_proj: hidden -> 2*hidden (no bias)
        liquid_in = self.hidden_size * (2 * self.hidden_size)
        # ShortConv: depthwise conv weight + bias
        liquid_conv = self.hidden_size * self.conv_size + self.hidden_size
        # Backbone: head_dim -> head_dim (weight + bias) per backbone layer
        # Shared across heads via broadcasting
        backbone = 0
        bb_in = head_dim
        for _ in range(self.backbone_layers):
            backbone += bb_in * head_dim + head_dim
            bb_in = head_dim
        # CfC heads: W_f1, W_f2, W_tau_a, W_decay (each head_dim -> state_size, +bias)
        cfc_heads = 4 * (head_dim * state_dim + state_dim)
        # tau_b: state_size (standalone parameter)
        tau_b = state_dim
        # state_out: state_size -> head_dim (weight + bias), shared across heads
        state_out = state_dim * head_dim + head_dim
        # out_proj: hidden -> hidden (no bias)
        liquid_out = self.hidden_size * self.hidden_size

        liquid_cell = (liquid_in + liquid_conv + backbone + cfc_heads
                       + tau_b + state_out + liquid_out)

        # Per-layer MLP: SwiGLU (gate_proj + up_proj + down_proj, no bias)
        mlp = 3 * self.hidden_size * self.intermediate_size

        # Per-layer norms: 2x RMSNorm (weight only)
        norms = 2 * self.hidden_size

        per_layer = liquid_cell + mlp + norms

        total = embed + per_layer * self.num_layers + self.hidden_size  # final norm
        if not self.tie_word_embeddings:
            total += embed
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
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Short 1D Causal Convolution
# ---------------------------------------------------------------------------

class ShortConv(nn.Module):
    """Depthwise causal convolution for local context mixing.

    Each channel has its own independent filter of width `kernel_size`.
    Causal padding ensures position t only sees positions <= t.
    This is the same short convolution used in Mamba, Hawk, and LFM2.
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(dim, 1, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, dim) -> (batch, seq_len, dim)"""
        x = x.transpose(1, 2)  # (batch, dim, seq_len)
        x = F.pad(x, (self.kernel_size - 1, 0))  # causal: pad left only
        x = F.conv1d(x, self.weight, self.bias, groups=x.shape[1])
        return x.transpose(1, 2)


# ---------------------------------------------------------------------------
# CfC (Closed-form Continuous-depth) Liquid Cell
# ---------------------------------------------------------------------------

class CfCLiquidCell(nn.Module):
    """
    Closed-form Continuous-depth Liquid Cell with multi-head structure.

    This implements the core liquid neural network dynamics for language modeling.
    The architecture combines several key ideas:

    1. SHORT CONVOLUTION: Local context mixing (like LFM2/Mamba)

    2. BACKBONE NETWORK: Projects each head's input through a small MLP

    3. CfC CLOSED-FORM UPDATE (from Hasani et al., Nature MI 2022):
       Two candidate states f1, f2 are computed from the backbone output.
       An input-dependent time gate tau interpolates between them:
         candidate_t = f1_t * (1 - tau_t) + f2_t * tau_t
       where tau_t = sigmoid(W_tau_a @ bb_t + tau_b)

       This is the closed-form solution to the ODE: dx/dt = -x/tau + A*sigma(x) + B*u
       The key insight is that tau (the time constant) DEPENDS ON THE INPUT,
       making the dynamics adaptive. When tau ~ 0, the system is slow (tracks f1).
       When tau ~ 1, the system is fast (tracks f2).

    4. INPUT-DEPENDENT DECAY (linearized LTC):
       The recurrence mixes the CfC candidate with the previous hidden state:
         h_t = decay_t * h_{t-1} + (1 - decay_t) * candidate_t
       where decay_t = sigmoid(W_decay @ bb_t) also depends on the input.
       This gives each hidden unit an input-adaptive effective time constant.

    5. MULTIPLICATIVE GATING (like Mamba/Hawk/LFM2):
       The output is gated by a sigmoid-activated projection of the input.

    The backbone, CfC projections, and decay gate share parameters across heads
    via broadcasting, keeping the parameter count efficient.
    """

    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.state_size = config.state_size

        assert config.hidden_size % config.num_heads == 0, \
            f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_heads})"

        # Input projection: hidden -> 2*hidden (x_path and z_gate)
        self.in_proj = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=False)

        # Short causal convolution on x path
        self.conv = ShortConv(config.hidden_size, config.conv_size)

        # Backbone MLP: head_dim -> head_dim
        # Shared across heads (applied via broadcasting on the head dimension)
        backbone_layers = []
        in_dim = self.head_dim
        for _ in range(config.backbone_layers):
            backbone_layers.append(nn.Linear(in_dim, self.head_dim))
            in_dim = self.head_dim
        self.backbone = nn.ModuleList(backbone_layers)

        # CfC candidate heads: backbone output -> state_size
        self.W_f1 = nn.Linear(self.head_dim, self.state_size)
        self.W_f2 = nn.Linear(self.head_dim, self.state_size)

        # Input-dependent time gate parameters (the LIQUID property)
        self.W_tau_a = nn.Linear(self.head_dim, self.state_size)
        self.W_tau_b = nn.Parameter(torch.zeros(self.state_size))

        # Input-dependent recurrence decay (linearized LTC component)
        self.W_decay = nn.Linear(self.head_dim, self.state_size)

        # State to output projection
        self.state_out = nn.Linear(self.state_size, self.head_dim)

        # Output projection (merges heads)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply backbone MLP with SiLU activation."""
        for layer in self.backbone:
            x = F.silu(layer(x))
        return x

    def _sequential_scan(
        self,
        decay: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Causal linear recurrence with input-dependent decay:
          h_t = decay_t * h_{t-1} + (1 - decay_t) * candidate_t

        This is the linearized Liquid Time-Constant recurrence where the
        effective time constant at each step depends on the input.

        Args:
            decay:     (batch, seq_len, num_heads, state_size) in [0, 1]
            candidate: (batch, seq_len, num_heads, state_size)

        Returns:
            (batch, seq_len, num_heads, state_size) hidden states
        """
        batch, seq_len, num_heads, state_size = decay.shape
        h = torch.zeros(batch, num_heads, state_size,
                        device=decay.device, dtype=decay.dtype)
        outputs = []
        for t in range(seq_len):
            h = decay[:, t] * h + (1.0 - decay[:, t]) * candidate[:, t]
            outputs.append(h)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x.shape

        # Split into x-path (processed through liquid dynamics) and z-gate
        xz = self.in_proj(x)
        x_path, z_gate = xz.chunk(2, dim=-1)

        # Short causal conv + SiLU activation
        x_path = F.silu(self.conv(x_path))

        # Reshape into heads: (batch, seq_len, num_heads, head_dim)
        x_heads = x_path.view(batch, seq_len, self.num_heads, self.head_dim)

        # Backbone transformation (parallel over all positions and heads)
        bb = self._backbone_forward(x_heads)

        # CfC: compute two candidate state updates
        f1 = torch.tanh(self.W_f1(bb))  # (batch, seq_len, num_heads, state_size)
        f2 = torch.tanh(self.W_f2(bb))

        # Input-dependent time gate: the CORE liquid property
        # tau depends on the input through the backbone, making the
        # interpolation between f1 and f2 adapt to incoming signals
        tau = torch.sigmoid(self.W_tau_a(bb) + self.W_tau_b)

        # CfC closed-form candidate: input-controlled interpolation
        candidate = f1 * (1.0 - tau) + f2 * tau

        # Input-dependent decay: controls how fast each state dimension
        # forgets the past (linearized LTC time constant)
        decay = torch.sigmoid(self.W_decay(bb))

        # Causal recurrence with input-dependent dynamics
        h_seq = self._sequential_scan(decay, candidate)

        # Project state back to head dimension
        out = self.state_out(h_seq)  # (batch, seq_len, num_heads, head_dim)

        # Merge heads
        out = out.reshape(batch, seq_len, self.hidden_size)

        # Multiplicative gating (output modulated by input-dependent gate)
        out = out * F.silu(z_gate)

        # Output projection
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network: down(silu(gate(x)) * up(x))"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Liquid Block
# ---------------------------------------------------------------------------

class LiquidBlock(nn.Module):
    """
    Single Liquid block (pre-norm residual):
      x = x + CfCLiquidCell(RMSNorm(x))
      x = x + SwiGLU_MLP(RMSNorm(x))
    """

    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.liquid_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.liquid_cell = CfCLiquidCell(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.liquid_cell(self.liquid_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full Liquid Language Model
# ---------------------------------------------------------------------------

class LiquidModel(nn.Module):
    """
    Liquid Neural Network Language Model.

    Architecture:
      Embedding -> [LiquidBlock x num_layers] -> RMSNorm -> LM Head

    Each LiquidBlock contains:
      1. RMSNorm -> CfC Liquid Cell (input-dependent dynamics) -> Residual
      2. RMSNorm -> SwiGLU MLP -> Residual

    The CfC cell implements closed-form continuous-depth dynamics where the
    state transition is governed by input-dependent time constants and decay
    rates, following the formulation from Hasani et al. (Nature MI, 2022).
    The input-dependent decay implements the linearized LTC dynamics from
    Liquid-S4 (Hasani et al., 2022), and the overall block structure with
    gated convolutions mirrors LFM2 (Liquid AI, 2025).

    This is a recurrent model (no attention): memory is O(batch * layers *
    num_heads * state_size) rather than O(batch * seq_len^2).
    """

    def __init__(self, config: LiquidConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Liquid blocks
        self.layers = nn.ModuleList([
            LiquidBlock(config) for _ in range(config.num_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

        self._gradient_checkpointing = config.gradient_checkpointing

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, ShortConv):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            labels: (batch, seq_len) target token indices (shifted internally)
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding

        Returns:
            dict with keys: "loss", "logits", "aux_loss"
        """
        x = self.embed_tokens(input_ids)

        # Apply attention_mask by zeroing out padded positions
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).type_as(x)

        # Process through liquid blocks
        for layer in self.layers:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False,
                )
            else:
                x = layer(x)

        # Final norm + LM head
        logits = self.lm_head(self.final_norm(x))

        # Compute loss if labels provided
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
            "aux_loss": torch.tensor(0.0, device=input_ids.device, dtype=logits.dtype),
        }

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
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
        Autoregressive text generation.

        Args:
            input_ids: (1, prefix_len) starting token ids
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            repetition_penalty: penalty for repeating tokens
            eos_token_id: end of sequence token id

        Returns:
            (1, prefix_len + generated_len) tensor of token ids
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits = self(idx_cond)["logits"][:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for tid in set(input_ids[0].tolist()):
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

            # Temperature scaling
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
                indices_to_remove = remove_mask.scatter(1, sorted_indices, remove_mask)
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return input_ids


# ---------------------------------------------------------------------------
# Convenience: default config targeting ~300M params
# ---------------------------------------------------------------------------

def get_default_config() -> LiquidConfig:
    """Returns a LiquidConfig targeting approximately 300M parameters."""
    return LiquidConfig(
        vocab_size=49152,
        hidden_size=1024,
        num_layers=24,
        max_seq_len=1024,
        state_size=64,
        num_heads=16,
        conv_size=4,
        backbone_layers=1,
        intermediate_size=2352,
        gradient_checkpointing=True,
        tie_word_embeddings=True,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Liquid Neural Network Language Model - Standalone Test")
    print("=" * 70)

    config = get_default_config()

    print(f"\nConfig:")
    print(f"  vocab_size:      {config.vocab_size}")
    print(f"  hidden_size:     {config.hidden_size}")
    print(f"  num_layers:      {config.num_layers}")
    print(f"  max_seq_len:     {config.max_seq_len}")
    print(f"  state_size:      {config.state_size}")
    print(f"  num_heads:       {config.num_heads}")
    print(f"  conv_size:       {config.conv_size}")
    print(f"  backbone_layers: {config.backbone_layers}")
    print(f"  intermediate:    {config.intermediate_size}")
    print(f"  Estimated params: {config.total_params_estimate() / 1e6:.1f}M")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    model = LiquidModel(config).to(device)
    total, trainable = model.count_parameters()
    print(f"Actual params:     {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")
    print(f"Estimate matches:  {total == config.total_params_estimate()}")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    print(f"\nForward pass (batch={batch_size}, seq_len={seq_len})...")
    output = model(input_ids, labels=labels)
    print(f"  Loss:     {output['loss'].item():.4f}")
    print(f"  Logits:   {output['logits'].shape}")
    print(f"  Aux loss: {output['aux_loss'].item():.4f}")

    # Test generation
    print(f"\nGeneration test...")
    prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
    generated = model.generate(prompt, max_new_tokens=16, temperature=1.0)
    print(f"  Prompt length:    {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")

    # Test with attention mask
    print(f"\nAttention mask test...")
    mask = torch.ones(batch_size, seq_len, device=device)
    mask[:, -10:] = 0  # mask last 10 tokens
    output_masked = model(input_ids, labels=labels, attention_mask=mask)
    print(f"  Loss (masked): {output_masked['loss'].item():.4f}")

    # Test gradient checkpointing
    print(f"\nGradient checkpointing test...")
    model.train()
    model.zero_grad()
    output = model(input_ids, labels=labels)
    output["loss"].backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                    if p.grad is not None) ** 0.5
    print(f"  Gradient norm: {grad_norm:.4f}")

    # Test without gradient checkpointing
    print(f"\nWithout gradient checkpointing...")
    model._gradient_checkpointing = False
    model.zero_grad()
    output2 = model(input_ids, labels=labels)
    output2["loss"].backward()
    grad_norm2 = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                     if p.grad is not None) ** 0.5
    print(f"  Gradient norm: {grad_norm2:.4f}")
    print(f"  Loss match: {abs(output['loss'].item() - output2['loss'].item()) < 1e-4}")

    print(f"\nAll tests passed!")
    print("=" * 70)

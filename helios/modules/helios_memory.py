"""Evolving-memory encoder for Helios (Echo-Infinity port).

Spec: docs/echo-to-helios-migration-design.md — chapter 1 defines the module
(parameters, gate, fp32 state contract), chapter 3 D3 makes FiLM(sigma_last)
conditioning of the write source mandatory (the three inference modes emit
the capture at sigma ~0.001 / ~0.02 / ~0.5), chapter 2 D4 requires token-level
frame masks for the partial first eviction write.

The module mirrors Echo model/query_memory.py with three adaptations:
ctx k/v projections (the Helios write source is last-block hidden states,
not cached attention K/V), FiLM(sigma_last) on the write source, and
frame masks. The rolling state `query_state` [B, M, dim] lives in fp32 as a
plain Python attribute: it must never enter state_dict (checkpoints carry
only `query_init` and weights), and fp32 fusion bounds recursive drift over
hundreds of writes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryCrossAttentionLayer(nn.Module):
    """One Enc layer: RMSNorm -> Q proj (+Q norm) -> cross-attn -> O + residual
    -> RMSNorm -> FFN(GELU-tanh) + residual. Ported from Echo
    query_memory.py:8-34; K/V arrive pre-projected from the write source."""

    def __init__(self, dim, num_heads, head_dim, ffn_mult):
        super().__init__()
        inner = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.q = nn.Linear(dim, inner)
        self.norm_q = nn.RMSNorm(inner, eps=1e-6)
        self.o = nn.Linear(inner, dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_mult * dim, dim),
        )

    def forward(self, state, k, v, attn_mask=None):
        # state [B, M, dim]; k/v [B, E, heads, head_dim]; attn_mask broadcastable
        # to [B, heads, M, E], True = attend.
        batch, m, _ = state.shape
        q = self.norm_q(self.q(self.norm1(state)))
        q = q.view(batch, m, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask
        )
        out = out.transpose(1, 2).reshape(batch, m, -1)
        state = state + self.o(out)
        state = state + self.ffn(self.norm2(state))
        return state


class HeliosMemoryEncoder(nn.Module):
    """Learnable evolving memory: fixed-shape query state refreshed from
    evicted-frame hidden states through cross-attention and a write gate."""

    def __init__(
        self,
        dim=5120,
        num_heads=40,
        head_dim=128,
        n_layers=2,
        ffn_mult=4,
        n_mem_frames=3,
        mem_frame_hw=(15, 26),
        gate_init_bias=0.75,
        initializer_range=0.014,
        sigma_cond=True,
    ):
        super().__init__()
        assert num_heads * head_dim == dim, "inner attention dim must equal dim"
        self.num_mem_tokens = n_mem_frames * mem_frame_hw[0] * mem_frame_hw[1]
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.query_init = nn.Parameter(
            torch.randn(1, self.num_mem_tokens, dim) * initializer_range
        )
        # Write source is hidden states, not cached K/V (unlike Echo), so the
        # encoder owns its context projections.
        self.ctx_k_proj = nn.Linear(dim, dim)
        self.ctx_v_proj = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(
            MemoryCrossAttentionLayer(dim, num_heads, head_dim, ffn_mult)
            for _ in range(n_layers)
        )
        # connector: Linear -> GELU -> Linear -> RMSNorm (Echo query_memory.py:85)
        self.connector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
            nn.RMSNorm(dim, eps=1e-6),
        )
        # Zero weight => the gate starts input-independent at sigmoid(bias);
        # bias is the retention knob the design retunes (0.75 vs Echo's 2.0).
        self.gate_linear = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, gate_init_bias)

        if sigma_cond:
            # FiLM on the write source, zero-init to exact identity.
            hidden = max(dim // 8, 8)
            self.sigma_mlp = nn.Sequential(
                nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 2 * dim)
            )
            nn.init.zeros_(self.sigma_mlp[-1].weight)
            nn.init.zeros_(self.sigma_mlp[-1].bias)
        else:
            self.sigma_mlp = None

        # Rolling state: plain attribute, fp32, NEVER registered (stays out of
        # state_dict / EMA / save hooks by construction).
        self.query_state = None

    def reset(self, batch_size, device=None):
        """Start a rollout: state := M0. Kept in the autograd graph so Stage-A
        training reaches query_init through reads."""
        init = self.query_init.float()
        if device is not None:
            init = init.to(device)
        self.query_state = init.expand(batch_size, -1, -1)

    def get_tokens(self, dtype=None):
        assert self.query_state is not None, "call reset() before get_tokens()"
        return self.query_state.to(dtype or self.query_init.dtype)

    def detach_state(self):
        """BPTT truncation. Contract (design ch.2 D7): call AFTER the read
        backward that consumes the latest write, BEFORE the next write."""
        if self.query_state is not None:
            self.query_state = self.query_state.detach()

    def update(self, evicted_hidden, frame_mask=None, sigma_last=None):
        """Gated write of one eviction event.

        evicted_hidden: [B, E, dim] last-block hidden states of the frames
            leaving the history window (already detached by the caller —
            gradients reach the encoder through the state -> read path).
        frame_mask: [B, E] bool, True = valid token. Required semantics for
            the first real eviction (design ch.2 D4: 1-8 valid frames).
        sigma_last: scalar or [B] tensor — the sigma of the capture forward.
        """
        assert self.query_state is not None, "call reset() before update()"
        batch, seq, _ = evicted_hidden.shape
        compute_dtype = self.query_init.dtype
        ctx = evicted_hidden.to(compute_dtype)

        if self.sigma_mlp is not None and sigma_last is not None:
            sigma = torch.as_tensor(
                sigma_last, dtype=compute_dtype, device=ctx.device
            ).reshape(-1, 1)
            scale, shift = self.sigma_mlp(sigma).chunk(2, dim=-1)
            ctx = ctx * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        k = self.ctx_k_proj(ctx).view(batch, seq, self.num_heads, self.head_dim)
        v = self.ctx_v_proj(ctx).view(batch, seq, self.num_heads, self.head_dim)

        attn_mask = None
        if frame_mask is not None:
            assert frame_mask.any(dim=1).all(), (
                "update() with an all-masked sample; skip the write instead"
            )
            attn_mask = frame_mask.view(batch, 1, 1, seq)

        state = self.query_state.to(compute_dtype)
        for layer in self.layers:
            state = layer(state, k, v, attn_mask)
        projected = self.connector(state)

        # fp32 gate fusion (design ch.1 §1): the recursion old -> new happens
        # in fp32 regardless of the module compute dtype.
        old32 = self.query_state.float()
        proj32 = projected.float()
        gate_in = torch.cat([old32.to(compute_dtype), projected], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_in)).float()
        self.query_state = gate * old32 + (1.0 - gate) * proj32

# K0 Memory Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/echo-to-helios-migration-design.md` — chapter 1 (module §1/§5/§7), chapter 2 (D2/D4/D13), chapter 3 (D3 FiLM, D15 detach contract), chapter 4 (R0/D1 arbitration, R2, K0).

**Goal:** Land the K0 pre-training foundations that need no GPU: the `HeliosMemoryEncoder` module, its unit-test suite (including the K0-mandated Enc/gate gradient-nonzero and conversion-lite tests), and the evolving-memory config flags + cross-flag assertion function wired into the trainer's validation block.

**Architecture:** A self-contained `nn.Module` (2-layer cross-attention encoder + sigmoid write gate over a fixed-shape fp32 query state held as a plain attribute), mirroring Echo `model/query_memory.py` with the three Helios adaptations from the design: ctx k/v projections (write source is hidden states, not cached KV), FiLM(σ_last) conditioning on the write source, and token-level frame masks for partial writes. Config assertions live in `helios/utils/train_config.py` as a pure function so they are unit-testable without running the 2761-line trainer.

**Tech Stack:** torch 2.10 (CPU is enough at test dims), stdlib `unittest` (pytest is absent from the team env `/mnt/beegfs/yuheng/miniconda3/envs/helios` and we do not install into it).

## Global Constraints

- Python for all commands: `PY=/mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python`, run from repo root with `PYTHONPATH=.`
- Code comments in English; commit messages English Conventional Commits (user rule).
- No modification to `transformer_helios.py` / pipelines in this plan — module registration (design ch.1 D9) is the next plan.
- `query_state` must be a plain attribute: never a Parameter/buffer, never in `state_dict` (design ch.1 §1).
- Gate weight zero-init + bias = `gate_init_bias` (0.75 default): gate starts input-independent at σ(bias) — makes the retention test deterministic and matches the design's "重标定 bias 为主旋钮" intent.
- FiLM output layer zero-init: σ-conditioning is exactly identity at init (no behavior change until trained).
- Deferred (documented, not implemented here): the D13 assertion `is_enable_evolving_memory ⇒ is_enable_stage1` — whether Stage-C self-forcing configs keep `is_enable_stage1` set is unverified against real configs; add it in the trainer-integration plan after checking `stage_3_post_self-forcing_version.yaml`.

## Smoke gate (repo convention)

Full-scale gate does not apply (no training). The gate for this plan: `PYTHONPATH=. $PY -m unittest discover -s tests -v` exits 0 with every test listed below passing, on the login node, CPU.

---

### Task 1: HeliosMemoryEncoder module + core behavior tests

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_helios_memory.py`
- Create: `helios/modules/helios_memory.py`

**Interfaces:**
- Produces (later plans rely on these exact signatures):
  - `HeliosMemoryEncoder.__init__(dim=5120, num_heads=40, head_dim=128, n_layers=2, ffn_mult=4, n_mem_frames=3, mem_frame_hw=(15, 26), gate_init_bias=0.75, initializer_range=0.014, sigma_cond=True)`
  - `reset(batch_size: int, device=None) -> None` — state := expanded `query_init` (graph kept: Stage A trains the static queries through reads)
  - `update(evicted_hidden: Tensor[B, E, dim], frame_mask: BoolTensor[B, E] | None = None, sigma_last: float | Tensor | None = None) -> None`
  - `get_tokens(dtype=None) -> Tensor[B, M, dim]` (default cast: module param dtype)
  - `detach_state() -> None`
  - `num_mem_tokens: int` attribute (M)

- [x] **Step 1: Write the failing core tests**

`tests/__init__.py`: empty file.

`tests/test_helios_memory.py`:

```python
"""Unit tests for HeliosMemoryEncoder (K0 foundations).

Design anchors: docs/echo-to-helios-migration-design.md ch.1 (module),
ch.2 D4 (frame mask), ch.3 D3/D15 (FiLM, detach contract), ch.4 K0
(gradient-nonzero + conversion tests). CPU-only, tiny dims.
"""

import unittest

import torch
import torch.nn as nn

from helios.modules.helios_memory import HeliosMemoryEncoder

DIM, HEADS, HEAD_DIM = 64, 4, 16
KW = dict(
    dim=DIM, num_heads=HEADS, head_dim=HEAD_DIM, n_layers=2, ffn_mult=2,
    n_mem_frames=2, mem_frame_hw=(2, 3), gate_init_bias=0.75,
)
B, E, M = 2, 20, 2 * 2 * 3


def make(seed=0, **overrides):
    torch.manual_seed(seed)
    return HeliosMemoryEncoder(**{**KW, **overrides})


class TestCoreBehavior(unittest.TestCase):
    def test_shapes_state_dtype_and_m0(self):
        mem = make()
        self.assertEqual(mem.num_mem_tokens, M)
        mem.reset(B)
        tokens = mem.get_tokens()
        self.assertEqual(tuple(tokens.shape), (B, M, DIM))
        # fp32 state invariant (design ch.1 §1)
        self.assertEqual(mem.query_state.dtype, torch.float32)
        # M0 always readable right after reset (design ch.3 D7)
        expected = mem.query_init.float().expand(B, -1, -1)
        self.assertTrue(torch.equal(mem.query_state, expected))

    def test_state_not_in_state_dict(self):
        mem = make()
        mem.reset(B)
        keys = mem.state_dict().keys()
        self.assertIn("query_init", keys)
        self.assertNotIn("query_state", keys)

    def test_update_gated_ema_formula(self):
        # With zero-init gate weight, g == sigmoid(bias) exactly:
        # new = g*old + (1-g)*projected (Echo query_memory.py:148-160 formula).
        mem = make()
        mem.reset(B)
        old = mem.query_state.clone()
        h = torch.randn(B, E, DIM)
        with torch.no_grad():
            ctx = h
            k = mem.ctx_k_proj(ctx).view(B, E, HEADS, HEAD_DIM)
            v = mem.ctx_v_proj(ctx).view(B, E, HEADS, HEAD_DIM)
            state = old.to(mem.query_init.dtype)
            for layer in mem.layers:
                state = layer(state, k, v, None)
            projected = mem.connector(state).float()
        mem.update(h)
        g = torch.sigmoid(torch.tensor(0.75))
        expected = g * old + (1 - g) * projected
        self.assertTrue(torch.allclose(mem.query_state, expected, atol=1e-5))

    def test_reset_clears_previous_rollout(self):
        mem = make()
        mem.reset(B)
        mem.update(torch.randn(B, E, DIM))
        after_write = mem.query_state.clone()
        mem.reset(B)
        self.assertFalse(torch.equal(mem.query_state, after_write))
        self.assertTrue(
            torch.equal(mem.query_state, mem.query_init.float().expand(B, -1, -1))
        )

    def test_update_before_reset_raises(self):
        mem = make()
        with self.assertRaises(AssertionError):
            mem.update(torch.randn(B, E, DIM))


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run tests to verify they fail on import**

Run: `cd /mnt/beegfs/siyuan/workspace/helios-echo && PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -m unittest tests.test_helios_memory -v`
Expected: `ModuleNotFoundError: No module named 'helios.modules.helios_memory'`

- [x] **Step 3: Implement the module**

`helios/modules/helios_memory.py`:

```python
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
```

- [x] **Step 4: Run the core tests, verify they pass**

Run: `cd /mnt/beegfs/siyuan/workspace/helios-echo && PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -m unittest tests.test_helios_memory -v`
Expected: 5 tests, `OK`

- [x] **Step 5: Commit**

```bash
git add helios/modules/helios_memory.py tests/__init__.py tests/test_helios_memory.py
git commit -m "feat(memory): add HeliosMemoryEncoder module skeleton with core tests"
```

---

### Task 2: K0 gradient/detach/mask/FiLM/conversion tests

**Files:**
- Modify: `tests/test_helios_memory.py` (append classes)

**Interfaces:**
- Consumes: Task 1 module exactly as defined.

- [x] **Step 1: Append the K0 test classes**

Append to `tests/test_helios_memory.py` (before the `__main__` block):

```python
class TestK0Gradients(unittest.TestCase):
    """K0 acceptance (design ch.4 K0/R2): Enc + gate + query_init must all
    receive nonzero gradient from a read that follows a write."""

    def _one_write_one_read_loss(self, mem):
        mem.reset(B)
        # Caller-detached write source (design ch.3 D15).
        mem.update(torch.randn(B, E, DIM), sigma_last=0.5)
        readout = nn.Linear(DIM, 1)
        return readout(mem.get_tokens()).sum()

    def test_enc_gate_query_grads_nonzero(self):
        mem = make(seed=1)
        loss = self._one_write_one_read_loss(mem)
        loss.backward()
        for name in [
            "ctx_k_proj.weight",
            "ctx_v_proj.weight",
            "layers.0.q.weight",
            "layers.1.ffn.0.weight",
            "connector.0.weight",
            "gate_linear.weight",
            "sigma_mlp.2.weight",
            "query_init",
        ]:
            param = dict(mem.named_parameters())[name]
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(param.grad.abs().max().item(), 0.0, name)

    def test_detach_state_cuts_graph(self):
        mem = make(seed=2)
        mem.reset(B)
        mem.update(torch.randn(B, E, DIM))
        mem.detach_state()
        loss = nn.Linear(DIM, 1)(mem.get_tokens()).sum()
        loss.backward()
        for _, param in mem.named_parameters():
            self.assertTrue(param.grad is None or param.grad.abs().max() == 0)

    def test_bptt1_second_window_still_learns(self):
        # detach AFTER read backward, BEFORE next write (D7 ordering): the
        # next write->read window must still produce Enc gradients.
        mem = make(seed=3)
        mem.reset(B)
        mem.update(torch.randn(B, E, DIM))
        nn.Linear(DIM, 1)(mem.get_tokens()).sum().backward()
        mem.detach_state()
        mem.zero_grad()
        mem.update(torch.randn(B, E, DIM))
        nn.Linear(DIM, 1)(mem.get_tokens()).sum().backward()
        grad = dict(mem.named_parameters())["ctx_k_proj.weight"].grad
        self.assertGreater(grad.abs().max().item(), 0.0)


class TestMaskAndFiLM(unittest.TestCase):
    def test_frame_mask_blocks_masked_tokens(self):
        mask = torch.zeros(B, E, dtype=torch.bool)
        mask[:, : E // 2] = True
        h = torch.randn(B, E, DIM)
        garbage = torch.randn(B, E, DIM)
        h_a = h.clone()
        h_b = h.clone()
        h_b[:, E // 2 :] = garbage[:, E // 2 :]

        mem_a, mem_b = make(seed=4), make(seed=4)
        mem_a.reset(B)
        mem_b.reset(B)
        mem_a.update(h_a, frame_mask=mask)
        mem_b.update(h_b, frame_mask=mask)
        self.assertTrue(torch.allclose(mem_a.query_state, mem_b.query_state, atol=1e-6))

    def test_all_masked_sample_raises(self):
        mem = make()
        mem.reset(B)
        with self.assertRaises(AssertionError):
            mem.update(torch.randn(B, E, DIM), frame_mask=torch.zeros(B, E, dtype=torch.bool))

    def test_film_identity_at_init(self):
        # Zero-init FiLM head: sigma conditioning is exactly identity at init.
        mem_a, mem_b = make(seed=5), make(seed=5)
        h = torch.randn(B, E, DIM)
        mem_a.reset(B)
        mem_b.reset(B)
        mem_a.update(h.clone(), sigma_last=0.5)
        mem_b.update(h.clone(), sigma_last=None)
        self.assertTrue(torch.equal(mem_a.query_state, mem_b.query_state))


class TestConversionLite(unittest.TestCase):
    """Checkpoint-conversion contract (design ch.4 D8(d), K0): same-config
    encoders interload strictly; shape mismatches fail loudly."""

    def test_same_config_strict_interload(self):
        src, dst = make(seed=6), make(seed=7)
        dst.load_state_dict(src.state_dict(), strict=True)
        for (name_a, param_a), (_, param_b) in zip(
            src.named_parameters(), dst.named_parameters()
        ):
            self.assertTrue(torch.equal(param_a, param_b), name_a)

    def test_mismatched_m_fails_loudly(self):
        src = make(seed=8)
        dst = make(seed=9, n_mem_frames=3)
        with self.assertRaises(RuntimeError):
            dst.load_state_dict(src.state_dict(), strict=True)
```

- [x] **Step 2: Run the full memory suite, verify pass**

Run: `cd /mnt/beegfs/siyuan/workspace/helios-echo && PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -m unittest tests.test_helios_memory -v`
Expected: 13 tests, `OK`

- [x] **Step 3: Commit**

```bash
git add tests/test_helios_memory.py
git commit -m "test(memory): add K0 gradient, detach, mask, FiLM and conversion tests"
```

---

### Task 3: Evolving-memory config flags + assertion function

**Files:**
- Modify: `helios/utils/train_config.py` (TrainingConfig tail ~L453; module-level function after the dataclasses)
- Modify: `train_helios.py` (config-validation block near the `# ---------------------- For Wan ----------------------` marker ~L2656; import line)
- Create: `tests/test_memory_config.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (independent).
- Produces: `validate_evolving_memory_config(training_config, data_config) -> None` (raises AssertionError), TrainingConfig fields exactly as in ch.2 D2 (training-only subset).

- [x] **Step 1: Write the failing config tests**

`tests/test_memory_config.py`:

```python
"""Cross-flag assertion tests for the evolving-memory config (design ch.2 D13)."""

import unittest

from helios.utils.train_config import (
    DataConfig,
    TrainingConfig,
    validate_evolving_memory_config,
)


def cfgs(**training_overrides):
    tc = TrainingConfig()
    dc = DataConfig()
    for key, value in training_overrides.items():
        assert hasattr(tc, key), key
        setattr(tc, key, value)
    return tc, dc


class TestMemoryConfigAssertions(unittest.TestCase):
    def test_defaults_pass(self):
        validate_evolving_memory_config(*cfgs())

    def test_train_without_enable_fails(self):
        tc, dc = cfgs(is_train_memory_module=True)
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)

    def test_memory_requires_multi_term_patch(self):
        tc, dc = cfgs(is_enable_evolving_memory=True, has_multi_term_memory_patch=False)
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)

    def test_dmd_requires_stage2_and_4_sections(self):
        tc, dc = cfgs(
            is_enable_evolving_memory=True,
            has_multi_term_memory_patch=True,
            is_train_dmd=True,
            is_enable_stage2=True,
            dmd_num_latent_sections_min=3,
        )
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)
        tc.dmd_num_latent_sections_min = 4
        validate_evolving_memory_config(tc, dc)
        tc.is_enable_stage2 = False
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)

    def test_dmd_rejects_gt_history(self):
        tc, dc = cfgs(
            is_enable_evolving_memory=True,
            has_multi_term_memory_patch=True,
            is_train_dmd=True,
            is_enable_stage2=True,
            dmd_num_latent_sections_min=4,
            is_use_gt_history=True,
        )
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)

    def test_tf_unroll_needs_stage1_dataset_and_sections(self):
        tc, dc = cfgs(
            is_enable_evolving_memory=True,
            has_multi_term_memory_patch=True,
            memory_tf_unroll=True,
            memory_unroll_sections=4,
        )
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)  # dataset not stage1
        dc.use_stage1_dataset = True
        validate_evolving_memory_config(tc, dc)
        tc.memory_unroll_sections = 1
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run to verify failure**

Run: `cd /mnt/beegfs/siyuan/workspace/helios-echo && PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -m unittest tests.test_memory_config -v`
Expected: `ImportError: cannot import name 'validate_evolving_memory_config'`

- [x] **Step 3: Add TrainingConfig fields + assertion function**

In `helios/utils/train_config.py`, append inside `TrainingConfig` after `clean_buffer_update_prob` (~L453):

```python
    # ---- Evolving memory (Echo-Infinity port) ----
    # Design: docs/echo-to-helios-migration-design.md ch.2 D2/D13. Arch-shape
    # keys must also flow into the transformer construction kwargs when the
    # module lands (single-complete-dict rule, ch.2 D2).
    is_enable_evolving_memory: bool = field(default=False)
    memory_num_query_frames: int = field(default=3)
    memory_enc_num_layers: int = field(default=2)
    memory_gate_init_bias: float = field(default=0.75)
    is_amplify_memory: bool = field(default=False)
    is_train_memory_module: bool = field(default=False)
    memory_freeze_backbone: bool = field(default=False)
    memory_learning_rate: float = field(default=5.0e-5)
    memory_bptt_sections: int = field(default=1)
    memory_write_source: str = field(default="t0_hidden")
    memory_single_write_prob: float = field(default=0.0)
    memory_tf_unroll: bool = field(default=False)
    memory_unroll_sections: int = field(default=1)
```

After the dataclass definitions (module level, before `@dataclass class Args` or after it — pick module tail), add:

```python
def validate_evolving_memory_config(training_config, data_config):
    """Cross-flag assertions for the evolving-memory feature (design ch.2 D13).

    Pure function so it is unit-testable without running the trainer; a no-op
    unless the feature is enabled. Called from the train_helios.py config
    validation block.
    """
    tc, dc = training_config, data_config
    if tc.is_train_memory_module:
        assert tc.is_enable_evolving_memory, (
            "is_train_memory_module requires is_enable_evolving_memory"
        )
    if not tc.is_enable_evolving_memory:
        return
    assert tc.memory_bptt_sections >= 1, "memory_bptt_sections must be >= 1"
    assert tc.has_multi_term_memory_patch, (
        "evolving memory conditions on the multi-term patchified history"
    )
    if tc.memory_tf_unroll:
        assert dc.use_stage1_dataset and not dc.use_stage3_dataset, (
            "TF unroll consumes the stage-1 history-latents dataset"
        )
        assert tc.memory_unroll_sections >= 2, (
            "TF unroll needs >= 2 sections to produce a write->read window"
        )
        assert not tc.is_train_dmd and not tc.use_error_recycling, (
            "TF unroll is a flow-matching regime"
        )
    if tc.is_train_dmd:
        assert tc.is_enable_stage2, (
            "memory DMD calibration is bound to the stage-2 pyramid rollout; "
            "the stage-1 rollout path is dead code (utils_helios_post.py:679)"
        )
        assert not tc.is_use_gt_history, (
            "GT-history DMD asserts single-section rollouts; memory needs >= 4"
        )
        assert (
            tc.dmd_num_latent_sections_min is not None
            and tc.dmd_num_latent_sections_min >= 4
        ), (
            "first real eviction write lands at section 2 and its read benefit "
            "at section 3; shorter rollouts give Enc/gate zero gradient"
        )
```

In `train_helios.py`: extend the existing `from helios.utils.train_config import ...` line with `validate_evolving_memory_config`, and insert the call directly under the `# ---------------------- For Wan ----------------------` marker:

```python
    validate_evolving_memory_config(conf.training_config, conf.data_config)
```

- [x] **Step 4: Run config tests + full suite, verify pass**

Run: `cd /mnt/beegfs/siyuan/workspace/helios-echo && PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -m unittest discover -s tests -v`
Expected: 19 tests, `OK`

- [x] **Step 5: Sanity-check no behavior change for existing configs**

Run: `cd /mnt/beegfs/siyuan/workspace/helios-echo && /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -c "import ast, sys; tree = ast.parse(open('train_helios.py').read()); print('parse ok')"`
Expected: `parse ok` (the trainer itself needs GPUs; parse + the default-config no-op test above stand in for it)

- [x] **Step 6: Commit**

```bash
git add helios/utils/train_config.py train_helios.py tests/test_memory_config.py
git commit -m "feat(memory): add evolving-memory config flags and cross-flag assertions"
```

---

## Self-Review

- Spec coverage: ch.1 §1 module (Task 1), ch.3 D3 FiLM + D15 detach (Tasks 1-2), ch.2 D4 mask (Task 2), ch.4 K0 gradient + D8(d) conversion-lite (Task 2), ch.2 D2/D13 flags + assertions incl. `dmd_num_latent_sections_min>=4` (Task 3). Registration into the transformer (ch.1 D9) and the full cross-lineage conversion test on the real transformer are explicitly the NEXT plan (needs transformer_helios.py changes).
- Types consistent: `update(evicted_hidden, frame_mask, sigma_last)` used identically in module and tests; `validate_evolving_memory_config(tc, dc)` signature matches call site.
- No placeholders.

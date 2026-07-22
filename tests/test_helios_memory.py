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


if __name__ == "__main__":
    unittest.main()

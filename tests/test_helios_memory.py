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

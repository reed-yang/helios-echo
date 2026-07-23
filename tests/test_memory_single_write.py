"""Stage A single-write helper tests (design ch.2 D5, Task 3b).

Reuses the tiny-transformer factory (and its import-time SDPA patch) from
test_transformer_memory. Tiny geometry: latent_window_size=3,
history_sizes=[4,2,1] (window 7).
"""

import unittest

import torch

from tests.test_transformer_memory import DEVICE, DTYPE, make_model
from helios.utils.utils_helios_base import _memory_single_write

LATENT_WINDOW = 3
HISTORY_SIZES = [4, 2, 1]
B = 2


def write_inputs(seed=3):
    torch.manual_seed(seed)
    return dict(
        evicted_latents=torch.randn(B, 4, LATENT_WINDOW, 8, 8, device=DEVICE, dtype=DTYPE),
        evicted_history_latents=torch.randn(B, 4, sum(HISTORY_SIZES), 8, 8, device=DEVICE, dtype=DTYPE),
        x0_latents=torch.randn(B, 4, 1, 8, 8, device=DEVICE, dtype=DTYPE),
        prompt_embeds=torch.randn(B, 5, 32, device=DEVICE, dtype=DTYPE),
    )


def run_write(model, valid_frames):
    return _memory_single_write(
        transformer=model,
        memory_module=model.evolving_memory,
        evicted_valid_frames=valid_frames,
        latent_window_size=LATENT_WINDOW,
        history_sizes=HISTORY_SIZES,
        device=DEVICE,
        dtype=DTYPE,
        **write_inputs(),
    )


class MemorySingleWriteTest(unittest.TestCase):
    def test_mixed_validity_blends_per_sample(self):
        model = make_model(with_memory=True)
        mem = model.evolving_memory
        mem.reset(B, device=DEVICE)
        m0 = mem.query_state.detach().clone()

        written = run_write(model, valid_frames=[0, 2])

        self.assertEqual(written, 1)
        state = mem.query_state
        # Sample 0 (valid == 0) keeps M0; sample 1 was written.
        self.assertTrue(torch.allclose(state[0].detach(), m0[0]))
        self.assertFalse(torch.allclose(state[1].detach(), m0[1]))

    def test_all_zero_validity_is_a_state_no_op(self):
        # The forward still RUNS (DDP rank symmetry — review blocker on the
        # original early return: skipping the wrapped forward on one rank
        # desynchronizes the buffer-broadcast collectives); only the state
        # outcome is a no-op via the blend.
        model = make_model(with_memory=True)
        mem = model.evolving_memory
        mem.reset(B, device=DEVICE)
        m0 = mem.query_state.detach().clone()

        written = run_write(model, valid_frames=[0, 0])

        self.assertEqual(written, 0)
        self.assertTrue(torch.allclose(mem.query_state.detach(), m0))

    def test_write_path_gradient_reaches_encoder(self):
        model = make_model(with_memory=True)
        mem = model.evolving_memory
        mem.reset(B, device=DEVICE)

        run_write(model, valid_frames=[LATENT_WINDOW, LATENT_WINDOW])
        # A read after the write: gradients must flow through the in-graph
        # update into the encoder's context projections (design D5 gradient
        # path: write -> state -> read -> loss).
        tokens = mem.get_tokens(torch.float32)
        tokens.sum().backward()

        grad = mem.ctx_k_proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

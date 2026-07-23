"""Section-level flow-loss primitive tests (design ch.2 D7, Stage A Task 3a).

Reuses the tiny-transformer factory (and its import-time SDPA patch) from
test_transformer_memory.
"""

import unittest
from types import SimpleNamespace

import torch

from tests.test_transformer_memory import DTYPE, make_inputs, make_model
from helios.utils.utils_helios_base import _flow_loss_section


def make_args(weighting_scheme="none"):
    return SimpleNamespace(training_config=SimpleNamespace(weighting_scheme=weighting_scheme))


def section_kwargs(model, memory_tokens=None):
    inputs = make_inputs()
    inputs.pop("return_dict")
    noisy = inputs.pop("hidden_states")
    timesteps = inputs.pop("timestep")
    prompt = inputs.pop("encoder_hidden_states")
    torch.manual_seed(7)
    target = torch.randn(1, 4, 3, 8, 8, dtype=DTYPE)
    sigmas = torch.tensor([0.5])
    return dict(
        args=make_args(),
        transformer=model,
        prompt_embeds=prompt,
        noisy_model_input=noisy,
        sigmas=sigmas,
        timesteps=timesteps,
        target=target,
        memory_tokens=memory_tokens,
        **inputs,
    )


class FlowLossSectionTest(unittest.TestCase):
    def test_returns_loss_without_backward(self):
        model = make_model(with_memory=True)
        loss, model_pred = _flow_loss_section(**section_kwargs(model))
        self.assertEqual(loss.dim(), 0)
        self.assertIsNotNone(loss.grad_fn)  # graph intact: caller owns backward
        self.assertEqual(tuple(model_pred.shape), (1, 4, 3, 8, 8))
        # No backward happened inside: parameters have no grads yet.
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_memory_tokens_reach_the_forward(self):
        model = make_model(with_memory=True)
        loss_plain, _ = _flow_loss_section(**section_kwargs(model))
        model.evolving_memory.reset(1)
        tokens = model.evolving_memory.get_tokens(DTYPE)
        loss_mem, _ = _flow_loss_section(**section_kwargs(model, memory_tokens=tokens))
        self.assertNotEqual(float(loss_plain), float(loss_mem))

    def test_read_path_gradient_reaches_memory_params(self):
        model = make_model(with_memory=True)
        model.evolving_memory.reset(1)
        tokens = model.evolving_memory.get_tokens(DTYPE)
        loss, _ = _flow_loss_section(**section_kwargs(model, memory_tokens=tokens))
        loss.backward()
        grad = model.evolving_memory.query_init.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

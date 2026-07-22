"""Pipeline memory state-machine contract tests (training-side pipeline_helios).

Tests stage1_sample's capture contract and get_memory_state export with a stub
pipeline self (no VAE/text encoder needed). The full section-loop queue
discipline (push, k-2 eviction write, per-capture sigma) is exercised by the
GPU rollout smoke (tests/smoke_rollout.py) — it lives inside __call__ and
needs real weights to run end to end.
"""

import unittest
from types import SimpleNamespace

import torch

from helios.pipelines.pipeline_helios import HeliosPipeline
from tests.test_transformer_memory import B, DIM, L_NOISY, make_inputs, make_model


class _SchedStub:
    def step(self, noise_pred, t, latents, return_dict=False):
        return (latents,)


class _BarStub:
    def update(self):
        pass


def _stage1_kwargs(inputs, **overrides):
    kwargs = dict(
        latents=inputs["hidden_states"].clone(),
        prompt_embeds=inputs["encoder_hidden_states"],
        negative_prompt_embeds=None,
        timesteps=torch.tensor([500.0, 100.0]),
        guidance_scale=1.0,
        indices_hidden_states=inputs["indices_hidden_states"],
        indices_latents_history_short=inputs["indices_latents_history_short"],
        indices_latents_history_mid=inputs["indices_latents_history_mid"],
        indices_latents_history_long=inputs["indices_latents_history_long"],
        latents_history_short=inputs["latents_history_short"],
        latents_history_mid=inputs["latents_history_mid"],
        latents_history_long=inputs["latents_history_long"],
        attention_kwargs=None,
        device="cpu",
        transformer_dtype=inputs["hidden_states"].dtype,
        generator=None,
        progress_bar=_BarStub(),
    )
    kwargs.update(overrides)
    return kwargs


def _stub_self(model):
    return SimpleNamespace(
        transformer=model,
        scheduler=_SchedStub(),
        do_classifier_free_guidance=False,
        interrupt=False,
        _current_timestep=None,
    )


class TestStage1CaptureContract(unittest.TestCase):
    def test_capture_returns_triple_with_sigma(self):
        model = make_model(with_memory=True, seed=20)
        model.evolving_memory.reset(B)
        inputs = make_inputs()
        stub = _stub_self(model)
        ret = HeliosPipeline.stage1_sample(
            stub,
            **_stage1_kwargs(
                inputs,
                memory_tokens=model.evolving_memory.get_tokens(),
                capture_last_step=True,
            ),
        )
        latents, capture, sigma_last = ret
        self.assertEqual(latents.shape, inputs["hidden_states"].shape)
        self.assertEqual(tuple(capture.shape), (B, L_NOISY, DIM))
        self.assertFalse(capture.requires_grad)
        # last timestep 100.0 -> sigma_last 0.1
        self.assertAlmostEqual(sigma_last, 0.1, places=6)

    def test_no_capture_keeps_plain_return(self):
        model = make_model(with_memory=True, seed=21)
        inputs = make_inputs()
        stub = _stub_self(model)
        ret = HeliosPipeline.stage1_sample(stub, **_stage1_kwargs(inputs))
        self.assertTrue(torch.is_tensor(ret))


class TestMemoryStateExport(unittest.TestCase):
    def test_none_when_no_encoder_or_state(self):
        bare = SimpleNamespace(transformer=SimpleNamespace())
        self.assertIsNone(HeliosPipeline.get_memory_state(bare))
        model = make_model(with_memory=True, seed=22)
        stub = SimpleNamespace(transformer=model)
        self.assertIsNone(HeliosPipeline.get_memory_state(stub))

    def test_roundtrip_content(self):
        model = make_model(with_memory=True, seed=23)
        model.evolving_memory.reset(B)
        cap = torch.randn(B, L_NOISY, DIM)
        stub = SimpleNamespace(transformer=model, _memory_queue=[(3, cap, 0.02)])
        state = HeliosPipeline.get_memory_state(stub)
        self.assertEqual(state["M"].dtype, torch.float32)
        self.assertTrue(
            torch.allclose(state["M"], model.evolving_memory.query_state.detach().float().cpu())
        )
        idx, saved_cap, sig = state["queue"][0]
        self.assertEqual((idx, sig), (3, 0.02))
        self.assertTrue(torch.equal(saved_cap, cap))


if __name__ == "__main__":
    unittest.main()

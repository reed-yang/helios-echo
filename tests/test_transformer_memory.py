"""Integration tests: HeliosMemoryEncoder registered in HeliosTransformer3DModel.

Covers the transformer-memory-integration plan gates: registration/legacy-load,
off-path bitwise equivalence, read path, capture API, and the K0 end-to-end
gradient through the full transformer. Runs on CPU by monkeypatching
attn_varlen_func to its SDPA fallback (helios_kernels/attention_dispatch.py
keeps flash-attn first when CUDA is available; the same suite re-runs on GPU
with real kernels).
"""

import unittest

import torch
import torch.nn.functional as F

import helios.modules.transformer_helios as th
from helios.modules.transformer_helios import HeliosTransformer3DModel

_ORIG_ATTN = th.attn_varlen_func

# On GPU the real flash-attn kernels run and require half precision; on CPU we
# monkeypatch the SDPA fallback and stay fp32.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def _sdpa_attn(q, k, v, attention_mask=None):
    assert attention_mask is None, "tiny tests use the unrestricted non-NAViT path"
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


# Patch at IMPORT time (not setUpModule): sibling test modules import the tiny
# helpers from here and may run first under unittest discover — the patch must
# be in place regardless of module execution order.
if not torch.cuda.is_available():
    th.attn_varlen_func = _sdpa_attn


TINY = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=8,
    in_channels=4,
    out_channels=4,
    text_dim=32,
    freq_dim=32,
    ffn_dim=32,
    num_layers=2,
    rope_dim=(4, 2, 2),
    zero_history_timestep=True,
    has_multi_term_memory_patch=True,
    guidance_cross_attn=True,
)
MEM = dict(
    is_enable_evolving_memory=True,
    memory_num_query_frames=2,
    memory_frame_hw=(2, 2),
    is_amplify_memory=True,
)
B, DIM = 1, 16
M = 2 * 2 * 2  # n_mem_frames * mem_h * mem_w
L_NOISY = 3 * 4 * 4  # current chunk post-patch tokens


def make_model(with_memory, seed=0):
    torch.manual_seed(seed)
    kwargs = dict(TINY)
    if with_memory:
        kwargs.update(MEM)
    model = HeliosTransformer3DModel(**kwargs)
    # init_weights() zero-inits proj_out (weight AND bias), which makes a fresh
    # model's output identically zero — a constant function that blinds every
    # output-based assertion. Re-randomize so outputs depend on inputs.
    with torch.no_grad():
        model.proj_out.weight.normal_(std=0.02)
        model.proj_out.bias.normal_(std=0.02)
    model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model


def make_inputs(device=DEVICE, dtype=DTYPE):
    torch.manual_seed(42)
    return dict(
        hidden_states=torch.randn(B, 4, 3, 8, 8, device=device, dtype=dtype),
        timestep=torch.tensor([500.0], device=device),
        encoder_hidden_states=torch.randn(B, 5, 32, device=device, dtype=dtype),
        indices_hidden_states=torch.tensor([[20.0, 21.0, 22.0]], device=device),
        indices_latents_history_short=torch.tensor([[19.0]], device=device),
        indices_latents_history_mid=torch.tensor([[17.0, 18.0]], device=device),
        indices_latents_history_long=torch.tensor([[13.0, 14.0, 15.0, 16.0]], device=device),
        latents_history_short=torch.randn(B, 4, 1, 8, 8, device=device, dtype=dtype),
        latents_history_mid=torch.randn(B, 4, 2, 8, 8, device=device, dtype=dtype),
        latents_history_long=torch.randn(B, 4, 4, 8, 8, device=device, dtype=dtype),
        return_dict=False,
    )


class TestRegistration(unittest.TestCase):
    def test_submodule_and_amp_params_registered(self):
        model = make_model(with_memory=True)
        names = dict(model.named_parameters())
        self.assertIn("evolving_memory.query_init", names)
        self.assertIn("evolving_memory.ctx_k_proj.weight", names)
        self.assertIn("blocks.0.attn1.memory_key_scale", names)
        self.assertIn("blocks.1.attn1.memory_key_scale", names)
        self.assertEqual(model.evolving_memory.num_mem_tokens, M)

    def test_legacy_state_dict_loads_missing_only_memory_keys(self):
        base = make_model(with_memory=False, seed=1)
        mem = make_model(with_memory=True, seed=2)
        result = mem.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        for key in result.missing_keys:
            self.assertTrue(
                key.startswith("evolving_memory.") or "memory_key_scale" in key, key
            )
        self.assertTrue(any(k.startswith("evolving_memory.") for k in result.missing_keys))

    def test_scale_memory_near_neutral_at_init(self):
        model = make_model(with_memory=True)
        scale = model.blocks[0].attn1.get_scale_memory()
        self.assertTrue(torch.allclose(scale.float(), torch.full_like(scale.float(), 1.1619), atol=2e-2))


class TestForward(unittest.TestCase):
    def _paired_models(self):
        base = make_model(with_memory=False, seed=3)
        mem = make_model(with_memory=True, seed=4)
        mem.load_state_dict(base.state_dict(), strict=False)
        return base, mem

    def test_off_path_bitwise_equivalence(self):
        base, mem = self._paired_models()
        inputs = make_inputs()
        with torch.no_grad():
            out_base = base(**inputs)[0]
            out_mem = mem(**inputs)[0]
        self.assertTrue(torch.equal(out_base, out_mem))

    def test_memory_tokens_change_output_shape_preserved(self):
        _, mem = self._paired_models()
        inputs = make_inputs()
        mem.evolving_memory.reset(B)
        with torch.no_grad():
            out_plain = mem(**inputs)[0]
            out_read = mem(**inputs, memory_tokens=mem.evolving_memory.get_tokens())[0]
        self.assertEqual(out_plain.shape, out_read.shape)
        self.assertFalse(torch.allclose(out_plain, out_read, atol=1e-7))

    def test_capture_api_tuple_arity(self):
        _, mem = self._paired_models()
        inputs = make_inputs()
        mem.evolving_memory.reset(B)
        tokens = mem.evolving_memory.get_tokens()
        with torch.no_grad():
            ret_plain = mem(**inputs, memory_tokens=tokens)
            ret_cap = mem(**inputs, memory_tokens=tokens, capture_last_hidden=True)
        self.assertEqual(len(ret_plain), 2)
        self.assertEqual(len(ret_cap), 3)
        self.assertEqual(tuple(ret_cap[2].shape), (B, L_NOISY, DIM))
        self.assertFalse(ret_cap[2].requires_grad)

    def test_fp32_kept_scale_with_half_model(self):
        # Regression (P1 smoke): from_pretrained keeps memory_key_scale in fp32
        # while the model runs bf16; the amp branch must cast the scale, or the
        # key promotes to fp32 and attention rejects the q/k dtype mismatch.
        _, mem = self._paired_models()
        mem.to(torch.bfloat16)
        for block in mem.blocks:
            block.attn1.memory_key_scale.data = block.attn1.memory_key_scale.data.float()
        mem.evolving_memory.reset(B)
        inputs = {
            k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() and v.ndim == 5 else v)
            for k, v in make_inputs().items()
        }
        inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"].to(torch.bfloat16)
        with torch.no_grad():
            out = mem(**inputs, memory_tokens=mem.evolving_memory.get_tokens())[0]
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_memory_requires_history_and_t0(self):
        _, mem = self._paired_models()
        mem.evolving_memory.reset(B)
        tokens = mem.evolving_memory.get_tokens()
        with self.assertRaises(AssertionError):
            mem(
                hidden_states=torch.randn(B, 4, 3, 8, 8, device=DEVICE, dtype=DTYPE),
                timestep=torch.tensor([500.0], device=DEVICE),
                encoder_hidden_states=torch.randn(B, 5, 32, device=DEVICE, dtype=DTYPE),
                memory_tokens=tokens,
                return_dict=False,
            )


class TestK0EndToEndGradient(unittest.TestCase):
    def test_read_path_grads_reach_encoder_and_amp(self):
        mem = make_model(with_memory=True, seed=5)
        mem.train()
        inputs = make_inputs()
        mem.evolving_memory.reset(B)
        mem.evolving_memory.update(torch.randn(B, 12, DIM, device=DEVICE, dtype=DTYPE), sigma_last=0.5)
        out = mem(**inputs, memory_tokens=mem.evolving_memory.get_tokens())[0]
        out.float().pow(2).mean().backward()
        names = dict(mem.named_parameters())
        for name in [
            "evolving_memory.query_init",
            "evolving_memory.ctx_k_proj.weight",
            "evolving_memory.gate_linear.weight",
            "blocks.0.attn1.memory_key_scale",
            "blocks.1.attn1.memory_key_scale",
        ]:
            grad = names[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().max().item(), 0.0, name)


if __name__ == "__main__":
    unittest.main()

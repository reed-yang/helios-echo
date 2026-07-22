"""Trainer wiring tests: extra-components persistence and param-group split.

Covers the trainer-wiring plan gates: save/load_extra_components section 5
round trip (keys, equality, fail-loud on a memory-less model), and
build_transformer_param_groups role/lr semantics. Uses the tiny transformer
from test_transformer_memory; CPU-only (no attention forward involved).
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from helios.utils.utils_base import (
    build_transformer_param_groups,
    load_extra_components,
    save_extra_components,
)
from tests.test_transformer_memory import make_model


def make_args(memory=True):
    return SimpleNamespace(
        training_config=SimpleNamespace(
            is_enable_stage1=False,
            is_train_full_multi_term_memory_patchg=False,
            is_train_lora_multi_term_memory_patchg=False,
            is_train_full_patch_embedding=False,
            restrict_self_attn=False,
            is_train_restrict_lora=False,
            is_amplify_history=False,
            is_use_gan=False,
            is_enable_evolving_memory=memory,
        )
    )


class TestExtraComponents(unittest.TestCase):
    def test_save_load_roundtrip(self):
        src = make_model(with_memory=True, seed=10)
        dst = make_model(with_memory=True, seed=11)
        args = make_args()
        with tempfile.TemporaryDirectory() as tmp:
            save_extra_components(args, model=src, output_dir=tmp)
            path = Path(tmp) / "transformer_partial.pth"
            saved = torch.load(path, map_location="cpu")
            self.assertTrue(any(k.startswith("evolving_memory.") for k in saved))
            self.assertIn("blocks.0.attn1.memory_key_scale", saved)
            self.assertIn("blocks.1.attn1.memory_key_scale", saved)
            # query_state must never be persisted
            self.assertFalse(any("query_state" in k for k in saved))

            load_extra_components(args, dst, str(path))
        for (name, p_src), (_, p_dst) in zip(
            src.evolving_memory.named_parameters(), dst.evolving_memory.named_parameters()
        ):
            self.assertTrue(torch.equal(p_src, p_dst), name)
        for i in range(2):
            self.assertTrue(
                torch.equal(
                    src.blocks[i].attn1.memory_key_scale, dst.blocks[i].attn1.memory_key_scale
                )
            )

    def test_load_into_memoryless_model_fails_loudly(self):
        src = make_model(with_memory=True, seed=12)
        bare = make_model(with_memory=False, seed=13)
        args = make_args()
        with tempfile.TemporaryDirectory() as tmp:
            save_extra_components(args, model=src, output_dir=tmp)
            with self.assertRaises(AssertionError):
                load_extra_components(args, bare, str(Path(tmp) / "transformer_partial.pth"))

    def test_feature_off_saves_no_memory_keys(self):
        src = make_model(with_memory=True, seed=14)
        args = make_args(memory=False)
        with tempfile.TemporaryDirectory() as tmp:
            save_extra_components(args, model=src, output_dir=tmp)
            saved = torch.load(Path(tmp) / "transformer_partial.pth", map_location="cpu")
            self.assertEqual(len(saved), 0)


class TestParamGroups(unittest.TestCase):
    def test_memory_only_group(self):
        model = make_model(with_memory=True, seed=15)
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if "evolving_memory" in name or "memory_key_scale" in name:
                param.requires_grad = True
        groups = build_transformer_param_groups(model, base_lr=1e-4, memory_lr=5e-5)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["role"], "memory")
        self.assertEqual(groups[0]["lr"], 5e-5)

    def test_base_and_memory_groups_ordered(self):
        model = make_model(with_memory=True, seed=16)
        model.requires_grad_(False)
        model.proj_out.weight.requires_grad = True
        for name, param in model.named_parameters():
            if "evolving_memory" in name:
                param.requires_grad = True
        groups = build_transformer_param_groups(model, base_lr=1e-4, memory_lr=5e-5)
        self.assertEqual([g["role"] for g in groups], ["base", "memory"])
        self.assertEqual([g["lr"] for g in groups], [1e-4, 5e-5])
        self.assertEqual(len(groups[0]["params"]), 1)

    def test_no_memory_single_base_group(self):
        model = make_model(with_memory=False, seed=17)
        model.requires_grad_(False)
        model.proj_out.weight.requires_grad = True
        groups = build_transformer_param_groups(model, base_lr=1e-4, memory_lr=5e-5)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["role"], "base")

    def test_all_frozen_raises(self):
        model = make_model(with_memory=True, seed=18)
        model.requires_grad_(False)
        with self.assertRaises(AssertionError):
            build_transformer_param_groups(model, base_lr=1e-4, memory_lr=5e-5)


if __name__ == "__main__":
    unittest.main()

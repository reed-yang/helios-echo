"""Cross-flag assertion tests for the evolving-memory config (design ch.2 D13)."""

import unittest

from helios.utils.train_config import (
    DataConfig,
    TrainingConfig,
    ValidationConfig,
    validate_evolving_memory_config,
)


def cfgs(**training_overrides):
    tc = TrainingConfig()
    dc = DataConfig()
    for key, value in training_overrides.items():
        assert hasattr(tc, key), key
        setattr(tc, key, value)
    return tc, dc


def valid_cfgs(dataset="stage1", **training_overrides):
    """A memory-enabled config that passes every assertion; perturb one field per test."""
    tc, dc = cfgs(
        is_enable_evolving_memory=True,
        has_multi_term_memory_patch=True,
        is_enable_stage1=True,
        **training_overrides,
    )
    if dataset == "stage1":
        dc.use_stage1_dataset = True
    elif dataset == "stage3":
        dc.use_stage3_dataset = True
    return tc, dc


class TestMemoryConfigAssertions(unittest.TestCase):
    def test_defaults_pass(self):
        validate_evolving_memory_config(*cfgs())

    def test_valid_enabled_config_passes(self):
        validate_evolving_memory_config(*valid_cfgs())

    def test_train_without_enable_fails(self):
        tc, dc = cfgs(is_train_memory_module=True)
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)

    def test_memory_requires_multi_term_patch(self):
        tc, dc = cfgs(is_enable_evolving_memory=True, has_multi_term_memory_patch=False)
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)

    def test_memory_requires_stage1(self):
        tc, dc = valid_cfgs()
        tc.is_enable_stage1 = False
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)

    def test_dataset_must_be_one_hot(self):
        # both false (raw-mp4 dataset) -> reject
        tc, dc = valid_cfgs()
        dc.use_stage1_dataset = False
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)
        # both true (silently routes to stage3) -> reject
        tc, dc = valid_cfgs()
        dc.use_stage3_dataset = True
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)

    def test_validation_kv_cache_conflict(self):
        tc, dc = valid_cfgs()
        vc = ValidationConfig()
        vc.use_kv_cache = True
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc, vc)
        # kv-cache off -> passes
        vc.use_kv_cache = False
        validate_evolving_memory_config(tc, dc, vc)

    def test_dmd_requires_stage2_and_4_sections(self):
        tc, dc = valid_cfgs(
            dataset="stage3",
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
        tc, dc = valid_cfgs(
            dataset="stage3",
            is_train_dmd=True,
            is_enable_stage2=True,
            dmd_num_latent_sections_min=4,
            is_use_gt_history=True,
        )
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)

    def test_tf_unroll_needs_stage1_dataset_and_sections(self):
        # tf_unroll additionally forbids the stage3 dataset (stricter than one-hot)
        tc, dc = valid_cfgs(dataset="stage3", memory_tf_unroll=True, memory_unroll_sections=4)
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)  # tf unroll rejects stage3
        dc.use_stage3_dataset = False
        dc.use_stage1_dataset = True
        validate_evolving_memory_config(tc, dc)
        tc.memory_unroll_sections = 1
        with self.assertRaises(AssertionError):
            validate_evolving_memory_config(tc, dc)


if __name__ == "__main__":
    unittest.main()

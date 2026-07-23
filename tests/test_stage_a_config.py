"""Stage A config lineage and evolving-memory contract tests."""

import unittest
from pathlib import Path

from omegaconf import OmegaConf

from helios.utils.train_config import Args, validate_evolving_memory_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "scripts/training/configs/stage1_lora_mem368_A.yaml"


class TestStageAConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_config = OmegaConf.load(CONFIG_PATH)
        cls.config = OmegaConf.merge(
            OmegaConf.structured(Args),
            cls.raw_config,
        )

    def test_all_memory_settings_are_explicit(self):
        expected_keys = {
            "is_enable_evolving_memory",
            "memory_num_query_frames",
            "memory_enc_num_layers",
            "memory_gate_init_bias",
            "is_amplify_memory",
            "is_train_memory_module",
            "memory_freeze_backbone",
            "memory_learning_rate",
            "memory_bptt_sections",
            "memory_write_source",
            "memory_single_write_prob",
            "memory_tf_unroll",
            "memory_unroll_sections",
        }
        self.assertTrue(expected_keys <= set(self.raw_config.training_config))

    def test_memory_settings(self):
        tc = self.config.training_config
        self.assertTrue(tc.is_enable_evolving_memory)
        self.assertTrue(tc.is_train_memory_module)
        self.assertTrue(tc.is_amplify_memory)
        self.assertTrue(tc.memory_freeze_backbone)
        self.assertEqual(tc.memory_learning_rate, 5.0e-5)
        self.assertEqual(tc.memory_single_write_prob, 0.5)
        self.assertFalse(tc.memory_tf_unroll)
        self.assertEqual(tc.memory_unroll_sections, 1)
        self.assertEqual(tc.memory_bptt_sections, 1)

    def test_stage_a_is_a_fresh_run_on_the_merged_base(self):
        # Stage A cannot fork by checkpoint-resume: its trainable set (memory
        # only) differs from the ancestor's, so accelerate load_state would
        # crash on the optimizer state. The fork is a MERGED base
        # (transformer_model_name_or_path) + fresh 4000-step run from
        # global_step 0; resume "latest" is crash-resume within this run only.
        self.assertEqual(self.config.training_config.max_train_steps, 4000)
        self.assertEqual(self.config.training_config.resume_from_checkpoint, "latest")
        self.assertIn("_merged", self.config.model_config.transformer_model_name_or_path)
        self.assertEqual(self.config.model_config.subfolder, "transformer")
        # Run artifacts belong to THIS fork's owner, not the ancestor's tree.
        self.assertTrue(str(self.config.output_dir).startswith("/mnt/beegfs/siyuan/"))

    def test_evolving_memory_validation_passes(self):
        validate_evolving_memory_config(
            self.config.training_config,
            self.config.data_config,
            self.config.validation_config,
        )

    def test_stage_a_inherits_required_stage1_lineage(self):
        tc = self.config.training_config
        self.assertTrue(tc.is_enable_stage1)
        self.assertTrue(self.config.data_config.use_stage1_dataset)
        self.assertTrue(tc.has_multi_term_memory_patch)
        self.assertTrue(tc.zero_history_timestep)


if __name__ == "__main__":
    unittest.main()

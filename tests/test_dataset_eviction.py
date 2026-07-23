import inspect
import unittest

import torch

from helios.dataset.dataloader_history_latents_dist import BucketedFeatureDataset, collate_fn


class DatasetEvictionTest(unittest.TestCase):
    @staticmethod
    def _make_timeline(channels=2, real_frames=32, height=2, width=3):
        prefix = torch.zeros(channels, 19, height, width)
        real_values = torch.arange(1, real_frames + 1, dtype=torch.float32)
        real = real_values.view(1, real_frames, 1, 1).expand(channels, -1, height, width)
        return torch.cat([prefix, real], dim=1)

    def test_eviction_slices_and_valid_frames_for_first_five_sections(self):
        timeline = self._make_timeline()
        expected_valid_frames = [0, 0, 0, 8, 9]
        expected_evicted_values = [
            [0] * 9,
            [0] * 9,
            [0] * 9,
            [0] + list(range(1, 9)),
            list(range(9, 18)),
        ]
        expected_history_values = [
            [0] * 19,
            [0] * 19,
            [0] * 19,
            [0] * 19,
            [0] * 11 + list(range(1, 9)),
        ]

        for choice_idx in range(5):
            with self.subTest(choice_idx=choice_idx):
                evicted, history, valid_frames = BucketedFeatureDataset._compute_eviction(
                    timeline, timeline, choice_idx, latent_window_size=9, history_window_size=19
                )
                self.assertEqual(evicted.shape, (2, 9, 2, 3))
                self.assertEqual(history.shape, (2, 19, 2, 3))
                self.assertEqual(valid_frames, expected_valid_frames[choice_idx])
                self.assertEqual(evicted[0, :, 0, 0].tolist(), expected_evicted_values[choice_idx])
                self.assertEqual(history[0, :, 0, 0].tolist(), expected_history_values[choice_idx])

    def test_k3_evicted_history_left_pads_exactly_one_frame(self):
        timeline = torch.arange(1, 1 + 2 * 50, dtype=torch.float32).view(2, 50, 1, 1)

        _, history, _ = BucketedFeatureDataset._compute_eviction(
            timeline, timeline, 3, latent_window_size=9, history_window_size=19
        )

        self.assertEqual(history[:, :1].tolist(), torch.zeros(2, 1, 1, 1).tolist())
        self.assertTrue(torch.equal(history[:, 1:], timeline[:, :18]))

    def test_mixed_resolution_roles(self):
        # Low-res bucket samples condition on a FULL-res source timeline while the
        # write forward's X_Noisy must be at the sample's own bucket resolution
        # (adversarial-review finding on 7f422b6): evicted comes from the bucket
        # timeline, history from the source timeline.
        bucket_timeline = self._make_timeline(height=2, width=3)
        source_timeline = self._make_timeline(height=8, width=12)

        for choice_idx in (0, 3, 4):
            with self.subTest(choice_idx=choice_idx):
                evicted, history, _ = BucketedFeatureDataset._compute_eviction(
                    bucket_timeline,
                    source_timeline,
                    choice_idx,
                    latent_window_size=9,
                    history_window_size=19,
                )
                self.assertEqual(evicted.shape, (2, 9, 2, 3))
                self.assertEqual(history.shape, (2, 19, 8, 12))

    def test_flag_defaults_false_and_collate_handles_eviction_fields(self):
        parameter = inspect.signature(BucketedFeatureDataset.__init__).parameters["return_evicted_latent"]
        self.assertIs(parameter.default, False)

        sample = {
            "evicted_latents": torch.ones(1, 9, 1, 1),
            "evicted_history_latents": torch.ones(1, 19, 1, 1),
            "evicted_valid_frames": 8,
        }
        batch = collate_fn([sample, sample])

        self.assertEqual(batch["evicted_latents"].shape, (2, 1, 9, 1, 1))
        self.assertEqual(batch["evicted_history_latents"].shape, (2, 1, 19, 1, 1))
        self.assertEqual(batch["evicted_valid_frames"], [8, 8])


if __name__ == "__main__":
    unittest.main()

"""D6 unroll-parameter tests for the stage-1 dataset (design ch.2 D6, F3 fix)."""

import os
import pickle
import tempfile
import unittest

import torch

from helios.dataset.dataloader_history_latents_dist import BucketedFeatureDataset


def make_dataset_stub(**overrides):
    """A BucketedFeatureDataset with __init__ skipped (no folder scan)."""
    ds = object.__new__(BucketedFeatureDataset)
    ds.history_sizes = [16, 2, 1]
    ds.is_keep_x0 = True
    ds.return_all_vae_latent = False
    ds.return_evicted_latent = False
    ds.return_rollout_metadata = False
    ds.num_rollout_sections = 3
    ds.base_seed = 42
    ds._epoch = 0
    for key, value in overrides.items():
        assert hasattr(ds, key), key
        setattr(ds, key, value)
    return ds


def synthetic_vae_latent(sections=6, channels=16, window=9, height=4, width=4):
    return torch.randn(sections, channels, window, height, width)


class UnrollShapeTest(unittest.TestCase):
    def test_clean_all_shape_for_u1_and_u4(self):
        for u in (1, 4):
            with self.subTest(u=u):
                ds = make_dataset_stub(return_all_vae_latent=True, num_rollout_sections=u)
                _, _, _, clean_all, _, start, _ = ds.prepare_stage1_latent(
                    synthetic_vae_latent(), idx=0
                )
                self.assertEqual(clean_all.shape, (16, 19 + 9 * u, 4, 4))
                self.assertIsNotNone(start)

    def test_insufficient_sections_raises(self):
        ds = make_dataset_stub(return_all_vae_latent=True, num_rollout_sections=7)
        with self.assertRaises(ValueError):
            ds.prepare_stage1_latent(synthetic_vae_latent(sections=6), idx=0)


class SeededStartTest(unittest.TestCase):
    def test_start_section_idx_is_deterministic_per_sample(self):
        ds = make_dataset_stub(return_all_vae_latent=True, num_rollout_sections=2)
        latent = synthetic_vae_latent()
        starts = {ds.prepare_stage1_latent(latent, idx=5)[5] for _ in range(4)}
        self.assertEqual(len(starts), 1)
        # A different idx draws from a different per-sample stream.
        ds._epoch = 3
        again = ds.prepare_stage1_latent(latent, idx=5)[5]
        self.assertIsInstance(again, int)

    def test_choice_idx_first_draw_matches_legacy_behavior(self):
        # The shared per-sample generator's FIRST draw must be bit-identical to the
        # previous inline fresh-generator draw, or every historical (seed, epoch, idx)
        # -> choice_idx mapping silently shifts.
        ds = make_dataset_stub()
        latent = synthetic_vae_latent()
        for idx in (0, 7, 123):
            expected = torch.randint(
                0,
                latent.shape[0],
                (1,),
                generator=torch.Generator().manual_seed(ds.base_seed + ds._epoch * 1000000 + idx),
            ).item()
            got = ds.prepare_stage1_latent(latent, idx=idx)[4]
            self.assertEqual(got, expected)

    def test_default_path_returns_none_extras(self):
        ds = make_dataset_stub()
        _, _, _, clean_all, _, start, eviction = ds.prepare_stage1_latent(
            synthetic_vae_latent(), idx=0
        )
        self.assertIsNone(clean_all)
        self.assertIsNone(start)
        self.assertIsNone(eviction)


class RolloutFilterTest(unittest.TestCase):
    def _folder_with_cache(self, tmpdir, frames_list):
        samples = []
        buckets = {}
        for i, num_frame in enumerate(frames_list):
            key = (num_frame, 4, 4)
            samples.append(
                {
                    "uttid": f"s{i}",
                    "dataset_name": tmpdir.rstrip("/"),
                    "file_path": os.path.join(tmpdir, f"s{i}.pt"),
                    "bucket_key": key,
                    "num_frame": num_frame,
                    "height": 4,
                    "width": 4,
                }
            )
            buckets.setdefault(key, []).append(i)
        with open(os.path.join(tmpdir, "dataset_cache.pkl"), "wb") as f:
            pickle.dump({"samples": samples, "buckets": buckets}, f)

    def test_init_filters_short_samples_only_when_unrolling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 121 frames -> 3 sections; 165 -> 5 sections.
            self._folder_with_cache(tmpdir, [121, 121, 165])

            ds4 = BucketedFeatureDataset(
                [tmpdir], return_all_vae_latent=True, num_rollout_sections=4
            )
            self.assertEqual(len(ds4.samples), 1)
            self.assertEqual(ds4.samples[0]["num_frame"], 165)
            self.assertEqual(sum(len(v) for v in ds4.buckets.values()), 1)

            ds3 = BucketedFeatureDataset(
                [tmpdir], return_all_vae_latent=True, num_rollout_sections=3
            )
            self.assertEqual(len(ds3.samples), 3)

            ds_off = BucketedFeatureDataset([tmpdir], num_rollout_sections=4)
            self.assertEqual(len(ds_off.samples), 3)


if __name__ == "__main__":
    unittest.main()

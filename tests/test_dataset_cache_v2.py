import os
import pickle
import tempfile
import unittest
from unittest import mock

from helios.dataset.dataloader_history_latents_dist import BucketedFeatureDataset


class DatasetCacheV2Test(unittest.TestCase):
    FILENAMES = (
        "full_121_368_640.pt",
        "half_132_184_320.pt",
        "quarter_165_92_160.pt",
        "other_121_360_640.pt",
        "short_120_368_640.pt",
    )

    @staticmethod
    def _create_empty_feature_files(folder, filenames):
        for filename in filenames:
            open(os.path.join(folder, filename), "wb").close()

    @staticmethod
    def _normalized_metadata(dataset):
        uttids = {sample["uttid"] for sample in dataset.samples}
        bucket_members = {
            bucket_key: {dataset.samples[index]["uttid"] for index in indices}
            for bucket_key, indices in dataset.buckets.items()
        }
        return uttids, bucket_members

    def test_force_rebuild_matches_v2_cache_load_after_single_res_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            self._create_empty_feature_files(folder, self.FILENAMES)

            fresh = BucketedFeatureDataset(
                [folder],
                force_rebuild=True,
                single_res=True,
                single_height=368,
                single_width=640,
            )
            self.assertFalse(os.path.exists(os.path.join(folder, "dataset_cache_v2.pkl")))
            BucketedFeatureDataset(
                [folder],
                single_res=True,
                single_height=368,
                single_width=640,
            )
            cached = BucketedFeatureDataset(
                [folder],
                single_res=True,
                single_height=368,
                single_width=640,
            )

            expected_uttids = {"full", "half", "quarter"}
            self.assertEqual(self._normalized_metadata(fresh), self._normalized_metadata(cached))
            self.assertEqual(self._normalized_metadata(cached)[0], expected_uttids)

    def test_creates_schema_2_superset_without_touching_legacy_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            self._create_empty_feature_files(folder, self.FILENAMES)
            legacy_path = os.path.join(folder, "dataset_cache.pkl")
            legacy_sentinel = b"legacy-cache-must-remain-byte-identical"
            with open(legacy_path, "wb") as legacy_file:
                legacy_file.write(legacy_sentinel)

            BucketedFeatureDataset(
                [folder],
                single_res=True,
                single_height=368,
                single_width=640,
            )

            v2_path = os.path.join(folder, "dataset_cache_v2.pkl")
            with open(v2_path, "rb") as cache_file:
                payload = pickle.load(cache_file)
            with open(legacy_path, "rb") as legacy_file:
                self.assertEqual(legacy_file.read(), legacy_sentinel)

            self.assertEqual(payload["schema"], 2)
            self.assertEqual(set(payload), {"schema", "samples", "buckets"})
            self.assertEqual(
                {sample["uttid"] for sample in payload["samples"]},
                {"full", "half", "quarter", "other"},
            )
            self.assertEqual(sum(len(indices) for indices in payload["buckets"].values()), 4)

    def test_wrong_v2_schema_is_rebuilt_instead_of_trusted(self):
        with tempfile.TemporaryDirectory() as folder:
            self._create_empty_feature_files(folder, self.FILENAMES)
            v2_path = os.path.join(folder, "dataset_cache_v2.pkl")
            with open(v2_path, "wb") as cache_file:
                pickle.dump({"schema": 1, "samples": [], "buckets": {}}, cache_file)

            dataset = BucketedFeatureDataset([folder])

            with open(v2_path, "rb") as cache_file:
                payload = pickle.load(cache_file)
            self.assertEqual(payload["schema"], 2)
            self.assertEqual(len(dataset), 4)

    def test_unreadable_v2_payload_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as folder:
            self._create_empty_feature_files(folder, self.FILENAMES)
            v2_path = os.path.join(folder, "dataset_cache_v2.pkl")
            with open(v2_path, "wb") as cache_file:
                cache_file.write(b"incomplete-pickle")

            dataset = BucketedFeatureDataset([folder])

            with open(v2_path, "rb") as cache_file:
                payload = pickle.load(cache_file)
            self.assertEqual(payload["schema"], 2)
            self.assertEqual(len(dataset), 4)

    def test_non_mapping_v2_payload_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as folder:
            self._create_empty_feature_files(folder, self.FILENAMES)
            v2_path = os.path.join(folder, "dataset_cache_v2.pkl")
            with open(v2_path, "wb") as cache_file:
                pickle.dump(["not", "a", "schema", "payload"], cache_file)

            dataset = BucketedFeatureDataset([folder])

            with open(v2_path, "rb") as cache_file:
                payload = pickle.load(cache_file)
            self.assertEqual(payload["schema"], 2)
            self.assertEqual(len(dataset), 4)

    def test_second_construction_loads_v2_without_scanning_or_rewriting(self):
        with tempfile.TemporaryDirectory() as folder:
            self._create_empty_feature_files(folder, self.FILENAMES)
            BucketedFeatureDataset([folder])
            v2_path = os.path.join(folder, "dataset_cache_v2.pkl")
            initial_mtime_ns = os.stat(v2_path).st_mtime_ns

            with mock.patch.object(
                BucketedFeatureDataset,
                "_build_folder_metadata",
                side_effect=AssertionError("second construction must not scan"),
            ):
                cached = BucketedFeatureDataset([folder])

            self.assertEqual(len(cached), 4)
            self.assertEqual(os.stat(v2_path).st_mtime_ns, initial_mtime_ns)

    def test_single_res_368x640_keeps_full_half_and_quarter_buckets(self):
        with tempfile.TemporaryDirectory() as folder:
            self._create_empty_feature_files(folder, self.FILENAMES)

            dataset = BucketedFeatureDataset(
                [folder],
                single_res=True,
                single_height=368,
                single_width=640,
            )

            self.assertEqual(
                {sample["uttid"] for sample in dataset.samples},
                {"full", "half", "quarter"},
            )
            self.assertEqual(
                set(dataset.buckets),
                {(121, 368, 640), (132, 184, 320), (165, 92, 160)},
            )
            all_indices = sorted(index for indices in dataset.buckets.values() for index in indices)
            self.assertEqual(all_indices, list(range(len(dataset.samples))))


if __name__ == "__main__":
    unittest.main()

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


class DatasetCacheV2HardeningTest(unittest.TestCase):
    """Review follow-ups on be8ce09: torn-write self-heal breadth, structural
    validation, atomic publish, and read-only-dir tolerance."""

    FILENAMES = DatasetCacheV2Test.FILENAMES

    def _folder_with_files(self, folder):
        DatasetCacheV2Test._create_empty_feature_files(folder, self.FILENAMES)

    def test_exotic_corrupt_payloads_are_rebuilt_not_raised(self):
        # A torn concurrent write can unpickle into raises far beyond
        # UnpicklingError. Each corrupt variant must trigger a silent rebuild.
        corrupt_variants = {
            # protocol-0 garbage that raises UnicodeDecodeError inside load
            "unicode": b"(V\xff\xfe\x00.",
            # references a nonexistent attribute -> AttributeError
            "attribute": b"cos\nnope_this_does_not_exist\n.",
            # valid pickle, dict with schema 2 but missing keys
            "missing_keys": pickle.dumps({"schema": 2}),
            # valid pickle, schema 2 but wrong value types
            "wrong_types": pickle.dumps({"schema": 2, "samples": "oops", "buckets": []}),
            # bucket values not lists
            "bad_buckets": pickle.dumps(
                {"schema": 2, "samples": [], "buckets": {"k": "not-a-list"}}
            ),
            # first sample valid but a later one is not (samples[0]-only check gap)
            "mixed_samples": pickle.dumps(
                {"schema": 2, "samples": [{"uttid": "ok"}, "not-a-dict"], "buckets": {}}
            ),
        }
        for name, payload in corrupt_variants.items():
            with self.subTest(variant=name):
                with tempfile.TemporaryDirectory() as folder:
                    self._folder_with_files(folder)
                    v2_path = os.path.join(folder, "dataset_cache_v2.pkl")
                    with open(v2_path, "wb") as cache_file:
                        cache_file.write(payload)

                    dataset = BucketedFeatureDataset([folder])

                    self.assertEqual(len(dataset), 4)
                    with open(v2_path, "rb") as cache_file:
                        healed = pickle.load(cache_file)
                    self.assertEqual(healed["schema"], 2)

    def test_readonly_folder_degrades_to_in_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            self._folder_with_files(folder)
            os.chmod(folder, 0o555)
            try:
                dataset = BucketedFeatureDataset([folder])
                self.assertEqual(len(dataset), 4)
                self.assertFalse(os.path.exists(os.path.join(folder, "dataset_cache_v2.pkl")))
                # No stray tmp files left behind either.
                leftovers = [f for f in os.listdir(folder) if ".tmp." in f]
                self.assertEqual(leftovers, [])
            finally:
                os.chmod(folder, 0o755)

    def test_cache_publish_is_atomic_via_replace(self):
        # The published file must appear via os.replace of a tmp file, never a
        # direct truncating open of the final path.
        with tempfile.TemporaryDirectory() as folder:
            self._folder_with_files(folder)
            replaced = []
            real_replace = os.replace

            def spy_replace(src, dst):
                replaced.append((src, dst))
                return real_replace(src, dst)

            with mock.patch("os.replace", side_effect=spy_replace):
                BucketedFeatureDataset([folder])

            v2_path = os.path.join(folder, "dataset_cache_v2.pkl")
            self.assertTrue(os.path.exists(v2_path))
            self.assertEqual(len(replaced), 1)
            self.assertIn(".tmp.", replaced[0][0])
            self.assertEqual(replaced[0][1], v2_path)


if __name__ == "__main__":
    unittest.main()

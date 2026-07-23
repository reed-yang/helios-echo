# Upstream PR draft — stage-1 dataset fixes (epoch + metadata cache)

> PR title: `Fix stage-1 persistent-worker epochs and reusable metadata caching`
> Branch: `fix/stage1-dataset-epoch-and-cache-v2` (worktree `/mnt/beegfs/siyuan/workspace/helios-upstream-pr`), based on `team/mid_training_xiangbo` @ `34f5a99`; commits `73b2d47` + `0f27099`.
> **SUBMITTED 2026-07-23 (user-approved): https://github.com/Visko-Platform/helios-team/pull/1** (base `mid_training_xiangbo`, GitHub tip verified = local base `34f5a99`).

# Summary

This PR ports two independently validated stage-1 dataset fixes:

1. Keep the dataset epoch live in fork-started persistent workers so per-sample seeded augmentation changes across epochs as intended.
2. Replace the configuration-dependent legacy metadata cache with a validated schema-v2 superset cache so repeated stage-1 launches can reuse the file listing.

The changes are split into two commits and introduce focused stdlib `unittest` coverage. No training configuration is changed by this PR.

# Defect 1: persistent workers keep a stale epoch

## Mechanism

`BucketedFeatureDataset` stored `_epoch` as a plain integer. The current stage-1 lineage creates eight persistent DataLoader workers. Each fork-started worker holds the dataset object copied at worker creation, so subsequent `set_epoch()` calls in the main process update only the parent copy. The per-sample seed

```text
base_seed + epoch * 1_000_000 + idx
```

therefore continued to use the worker-creation epoch. In current stage-1 training this collapses the seeded `choice_idx` augmentation to the creation-time epoch instead of varying it across epochs.

## Impact

Samples remain loadable and training continues, but the intended cross-epoch augmentation is silently lost under the actual eight-worker persistent-loader configuration. The same sample index repeatedly draws from the creation-time epoch stream rather than the current epoch stream.

## Fix design

The dataset epoch is backed by `multiprocessing.Value("i", 0)` behind the existing integer-valued `_epoch` property. Fork descendants inherit the shared object, so a parent `set_epoch()` update becomes visible to already-running persistent workers. `__getstate__()` converts the shared value to a frozen integer for spawn-style pickling, preserving the previous non-shared behavior instead of failing serialization.

## Behavior-change analysis

This is an intended behavior change: stage-1 `choice_idx` augmentation once again varies across epochs under fork-started persistent workers. Single-process use retains the same public integer property and seed formula. Spawn-pickled copies receive the epoch value at serialization time and remain independent, matching the old plain-integer sharing behavior.

# Defect 2: every launch repeats the 388k-file scan

## Mechanism

The legacy `dataset_cache.pkl` payload contains metadata after configuration-dependent filtering but does not record the filter parameters. Because `single_res` could not safely trust such a cache, `train_helios.py` required `force_rebuild: true` whenever `single_res` was enabled. A stage-1 launch therefore re-listed and parsed roughly 388,000 BeeGFS filenames once per rank on every launch.

## Fix design

Stage-1 now uses `dataset_cache_v2.pkl`, whose payload is the configuration-independent superset:

```text
{"schema": 2, "samples": [...], "buckets": {...}}
```

Only the hardcoded `num_frame >= 121` rule applies during the directory scan. The `single_res` filter runs after load in memory, followed by a bucket rebuild with new contiguous sample indices.

Loaded payloads are accepted only when `_validate_cache_payload()` confirms schema 2, a list of sample dictionaries, and a dictionary whose bucket values are lists. Any load exception or invalid structure triggers a silent rebuild. First-build publication writes `dataset_cache_v2.pkl.tmp.<pid>` and uses `os.replace()` so readers do not observe a partially written final file. An `OSError` while saving prints `Cache save skipped ... continuing with in-memory metadata`, removes any temporary file when possible, and continues using the freshly scanned metadata.

`force_rebuild` remains an explicit full-rescan override: it performs no cache read and no cache write. The trainer assertion is retained only for non-stage1 datasets, which still use the legacy configuration-dependent cache.

The legacy `dataset_cache.pkl` is never read, written, renamed, or deleted; existing bytes remain untouched.

## Behavior-change analysis

The cache optimization is inert while existing stage-1 configurations keep `force_rebuild: true`: launches continue to rescan and do not publish or consume v2 caches. Changing a stage-1 configuration to `force_rebuild: false` explicitly opts into the new reusable cache. Resolution filtering produces the same in-memory sample and bucket membership as a fresh scan.

# Validation

The focused suite contains 14 stdlib `unittest` cases:

- shared epoch property update after construction;
- synchronized fork-worker observation of a post-fork epoch update;
- spawn-style pickle degradation to a frozen integer and independent subsequent updates;
- epoch-dependent `choice_idx` draws;
- fresh-scan versus v2-load parity under `single_res`;
- schema-v2 superset contents and byte-identical legacy-cache preservation;
- wrong-schema, unreadable, non-mapping, and structurally malformed payload self-healing;
- cache reuse without a second scan or rewrite;
- full/half/quarter resolution filtering with rebuilt indices;
- read-only-directory fallback without temporary-file debris;
- atomic publication through `os.replace()`.

Validation command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -m unittest tests.test_dataset_cache_v2 tests.test_dataset_epoch -v
```

Result: 14 tests passed. The same fixes have also been running in the `echo-memory` fork's 79-test suite.

# Notes

- A second plain-integer `_epoch` exists in the sibling `BucketedSampler` class near upstream line 407. It has the same visual pattern but a different ownership/use path and is intentionally not touched in this focused PR.
- Rank-0/barrier coordination for the first cache build is deferred. Concurrent first launches may still perform one scan per rank, but atomic publication prevents torn final cache files; subsequent launches reuse the published cache.
- The PR does not include caption-version pinning, rollout metadata, eviction logic, memory symbols, rollout-section changes, or any fork-only training configuration.

# Suggested config follow-up

After this PR is merged, set `force_rebuild: false` in stage-1 configurations that should reuse `dataset_cache_v2.pkl`. Keep `force_rebuild: true` only when an explicit fresh directory rescan is desired. Non-stage1 `single_res` datasets must continue to use `force_rebuild: true` until their legacy caches receive an equivalent configuration-independent schema.

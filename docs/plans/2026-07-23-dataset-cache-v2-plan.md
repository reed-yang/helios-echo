# Dataset Cache v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate repeated Stage-1 filename scans for `single_res` runs by caching a configuration-independent metadata superset in a versioned cache and filtering resolutions in memory.

**Architecture:** `BucketedFeatureDataset` exclusively reads/writes `dataset_cache_v2.pkl` with payload `{"schema": 2, "samples": [...], "buckets": {...}}`; it never reads or rewrites `dataset_cache.pkl`. Folder scanning retains only the configuration-independent `num_frame >= 121` predicate, while `single_res` and rollout-length predicates run after all folder metadata is loaded and rebuild contiguous bucket indices.

**Tech Stack:** Python stdlib (`os`, `pickle`, `tempfile`, `unittest`, `unittest.mock`), PyTorch dataset code.

## Global Constraints

- Preserve `_epoch` shared-memory behavior and `__getstate__`, eviction and rollout return paths, rollout-length filtering, and caption-version pinning.
- Keep `force_rebuild=True` as a scan override with no cache write.
- Do not read, trust, rewrite, or delete legacy `dataset_cache.pkl`.
- Do not add rank-0/barrier coordination; document that trainer-side coordination is deferred.
- Use English code comments, stdlib `unittest`, no package installation, and do not modify `tools/offload_data`.
- Do not commit unless explicitly requested.

---

### Task 1: Specify v2 cache behavior with tests

**Files:**
- Create: `tests/test_dataset_cache_v2.py`
- Modify: `tests/test_dataset_unroll.py:117-156`

**Interfaces:**
- Consumes: `BucketedFeatureDataset(feature_folders, force_rebuild=False, single_res=False, single_height=384, single_width=640)`.
- Produces: regression coverage for schema, legacy isolation, scan reuse, force-rebuild equivalence, and 368x640 full/half/quarter filtering.

- [ ] **Step 1: Add fake-filename fixtures and normalized metadata comparison**

Create empty `.pt` files named `<uttid>_<num_frame>_<height>_<width>.pt`. Normalize a dataset as `(set(uttids), {bucket_key: set(bucket_uttids)})` so fresh-scan and cache-load paths can be compared without depending on `os.listdir` order.

- [ ] **Step 2: Add the four required tests**

Test that: (a) `force_rebuild=True, single_res=True` and a normal v2 cache construction/load yield identical normalized samples and bucket membership; (b) v2 has exact schema `2`, includes the unfiltered resolution superset but excludes `<121` frames, and a byte-sentinel legacy cache is untouched; (c) a second construction succeeds while `_build_folder_metadata` is patched to raise, and leaves v2 mtime unchanged; (d) 368x640 filtering retains `(368,640)`, `(184,320)`, `(92,160)` and rejects other resolutions.

- [ ] **Step 3: Update the rollout filter fixture**

Write `dataset_cache_v2.pkl` with `{"schema": 2, "samples": samples, "buckets": buckets}` instead of writing a legacy cache, preserving the rollout test's purpose under the new reader contract.

- [ ] **Step 4: Run the new test to establish the red state**

Run:
```bash
PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -m unittest tests.test_dataset_cache_v2 -v
```
Expected before implementation: failures because v2 is not created/loaded and scan-time filtering prevents an unfiltered superset.

---

### Task 2: Implement versioned superset caching and in-memory filtering

**Files:**
- Modify: `helios/dataset/dataloader_history_latents_dist.py:60-164`
- Modify: `train_helios.py:2774-2775`

**Interfaces:**
- Consumes: `dataset_cache_v2.pkl` payload with integer `schema == 2` and existing sample/bucket structures.
- Produces: `_build_folder_metadata(folder)` independent of resolution configuration; `_process_folder(folder, cache_file)` that loads only valid v2 payloads or scans; post-load `single_res` filtering with contiguous bucket indices.

- [ ] **Step 1: Select only the v2 filename**

Change constructor cache selection to `os.path.join(folder, "dataset_cache_v2.pkl")`. Keep the legacy filename unreferenced so old readers and files remain unaffected.

- [ ] **Step 2: Validate schema or rebuild**

When `force_rebuild` is false and v2 exists, unpickle it and accept it only if `cached_data.get("schema") == 2`. Otherwise call `_build_folder_metadata`. Save `{"schema": 2, "samples": folder_samples, "buckets": folder_buckets}` only when `force_rebuild` is false. Add an English comment that rank-0/barrier coordination is intentionally deferred to trainer-side work.

- [ ] **Step 3: Make the scan configuration-independent**

Delete the `allowed_resolutions` calculation and `self.single_res` predicate from `_build_folder_metadata`; retain `if num_frame < 121: continue` unchanged.

- [ ] **Step 4: Apply `single_res` after all cache loads**

In `__init__`, compute allowed resolutions from configured full, half, and quarter dimensions, filter `self.samples` with a list comprehension, and rebuild `self.buckets` by enumerating the filtered list so every index is contiguous and valid. Keep this separate from and before the existing rollout-length filter.

- [ ] **Step 5: Remove the obsolete trainer assertion**

Delete the `single_res -> force_rebuild` assertion. Do not alter passing of `force_rebuild` into the dataset.

- [ ] **Step 6: Run focused dataset tests**

Run:
```bash
PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -m unittest tests.test_dataset_cache_v2 tests.test_dataset_eviction tests.test_dataset_unroll -v
```
Expected: all tests pass.

---

### Task 3: Verify and inspect the complete change

**Files:**
- Inspect: `helios/dataset/dataloader_history_latents_dist.py`
- Inspect: `train_helios.py`
- Inspect: `tests/test_dataset_cache_v2.py`
- Inspect: `tests/test_dataset_unroll.py`

**Interfaces:**
- Consumes: completed implementation and tests.
- Produces: focused/full-suite evidence and a clean scoped diff.

- [ ] **Step 1: Run the full suite**

Run:
```bash
PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python -m unittest discover -s tests -p "test_*.py"
```
Record the final `Ran ... tests` and `OK` lines.

- [ ] **Step 2: Inspect diff and required artifacts**

Run:
```bash
git diff --check
git diff --stat
ls -l tests/test_dataset_cache_v2.py
git status --short
```
Then inspect the targeted diff to confirm no formatting churn, legacy cache writes, rank coordination, or unrelated changes.

- [ ] **Step 3: Return the evidence card**

Report exact file:line edits, schema choice, force-rebuild/reference equivalence method, focused and full-suite results, diff stat, new test listing, and remaining risk that simultaneous first-time multi-rank cache construction is intentionally deferred.

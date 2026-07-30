# P2 Quality Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan task-by-task in the current session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one campaign-aware CLI that reuses the repository's existing perceptual metric adapters to score already-generated P2 videos with manifest-aligned prompts and resumable JSON output.

**Architecture:** The new script owns only P2 discovery, manifest prompt normalization, metric availability validation, idempotent persistence, aggregation, and CLI reporting. It passes `common.EvalRow` objects into the unchanged adapters under `scripts/evaluation/metrics/`; each metric receives all compatibility/default arguments its existing adapter expects. Metric defaults are restricted to adapters proven importable, locally weighted, and offline-runnable in the shared environment.

**Tech Stack:** Python 3, argparse, pathlib, json, statistics, existing `common.py`/`metrics/*.py`, PyTorch/CUDA, Slurm `srun` for one bounded smoke allocation.

## Global Constraints

- Work only in `/mnt/beegfs/siyuan/workspace/helios-echo` on branch `echo-memory`.
- Create exactly one product script: `scripts/evaluation/score_p2_quality.py`; do not alter existing metric implementations or unrelated formatting.
- Do not install or download anything; use `/mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python`.
- Do not commit and do not submit Slurm jobs.
- Use at most one direct `srun -p gpu -w c-node08 --gres=gpu:h200:1 --cpus-per-task=8 --mem=64G --time=00:20:00 ...` allocation and score at most two videos per requested campaign.
- Preserve exact manifest prompts. For prompt lists, score against the first item and record that selection in JSON.
- A requested unavailable metric must fail before scoring with a message naming its missing import or local artifact.
- Final user-facing handoff is at most 45 lines and includes availability evidence, smoke scores, full command, and long-video sampling/OOM caveats.

---

### Task 1: Establish Metric Availability and Offline Requirements

**Files:**
- Read: `scripts/evaluation/common.py`
- Read: `scripts/evaluation/run_helios_long_segment_metric.py`
- Read: `scripts/evaluation/metrics/{dover,hpsv3,pickscore,videoalign}_metric.py`
- Read/Search: `third-party/**`, team configs/launchers, Hugging Face caches, and likely checkpoint roots under `/mnt/beegfs`

**Interfaces:**
- Consumes: existing adapter defaults and constructor behavior.
- Produces: a four-row evidence matrix with required weights, local paths, clean-import result, and offline verdict; the passing names define `DEFAULT_METRICS` in Task 2.

- [ ] Inspect local third-party repositories/config defaults and find all explicit model/checkpoint paths.
- [ ] Search bounded likely roots and Hugging Face caches for each required artifact without downloading.
- [ ] Run isolated import probes in the shared environment with offline environment variables enabled.
- [ ] Run constructor/load probes only where local artifacts exist; record the first exact failure for unavailable metrics.
- [ ] Decide the supported/default metric subset strictly from local import + weights + offline evidence.

### Task 2: Implement P2 Campaign Discovery and Resumable Scoring

**Files:**
- Create: `scripts/evaluation/score_p2_quality.py`

**Interfaces:**
- Consumes: `common.EvalRow`, `common.add_third_party_paths`, and each passing `metrics.<name>_metric.score(rows, args, cache_root)` adapter.
- Produces: `main(argv=None) -> int`; one JSON file per `(campaign, metric)` at `<results-root>/<run_id>/quality/<metric>.json` unless `--out` overrides the destination template/root; final `QUALITY ...` lines and `EXIT_CODE=0`.

- [ ] Add argparse for repeatable required `--campaign`, `--results-root`, comma-separated `--metrics`, `--device`, `--limit`, `--out`, `--dry-run`, and `--force`, plus adapter options needed by supported metrics.
- [ ] Discover sorted `<campaign>/*/prompt_*.mp4` paths, require sibling manifests, parse string/list prompts, and build stable video keys relative to the campaign directory.
- [ ] Validate requested metric names and local requirements before any scoring; include actionable missing-artifact messages for all four known metrics.
- [ ] Load an existing metric JSON, skip keyed videos already carrying the metric's primary score unless `--force`, and pass only pending rows to the unchanged adapter.
- [ ] Treat adapter-populated row errors or absent/non-finite primary scores as a failed run rather than writing false success.
- [ ] Atomically persist per-video prompt metadata and score dictionaries after each scoring batch, then recompute finite mean/median over the primary score.
- [ ] Implement dry-run discovery without model loading or output writes, while printing the videos and normalized prompt choices.
- [ ] Print exactly one final `QUALITY campaign=<id> metric=<m> scored=<n>/<total> mean=<x>` per pair and then `EXIT_CODE=0`.

### Task 3: CPU/CLI Verification Before GPU Spend

**Files:**
- Verify: `scripts/evaluation/score_p2_quality.py`

**Interfaces:**
- Consumes: Task 2 CLI.
- Produces: syntax/import evidence, dry-run discovery evidence, and idempotence/output-schema evidence without invoking unavailable metrics.

- [ ] Run `python -m py_compile scripts/evaluation/score_p2_quality.py`.
- [ ] Run `--dry-run --limit 2` for both named campaigns and verify deterministic ordering, arm-qualified keys, exact manifests, and list-prompt choice recording.
- [ ] Request every unavailable metric once in dry-run/preflight mode and verify the error names the missing artifact/import without network access.
- [ ] Inspect the script diff for changes outside the single new script and this required plan.

### Task 4: One-Allocation GPU Smoke and Idempotence Check

**Files:**
- Execute: `scripts/evaluation/score_p2_quality.py`
- Inspect: generated ignored JSON under `results/p2_interim/<campaign>/quality/`

**Interfaces:**
- Consumes: supported metric list from Task 1 and the two requested campaigns.
- Produces: actual per-video scores for two videos per campaign for every working metric, summary lines, and rerun skip evidence.

- [ ] Before allocation, inspect c-node08 GPU process ownership and VRAM headroom without starting a job; abort rather than overlap a protected owner.
- [ ] Use one exact bounded `srun` allocation to run both campaigns with `--limit 2` and every supported metric, with offline environment variables set.
- [ ] Within the same allocation/command, rerun once without `--force` and verify `scored=0/<total>` while summaries remain unchanged.
- [ ] Read the generated JSON and capture campaign/video/primary-score values plus means/medians.
- [ ] Run final `git status` and `git diff --check`; verify no commit, no downloaded artifacts, and no unrelated tracked modifications.

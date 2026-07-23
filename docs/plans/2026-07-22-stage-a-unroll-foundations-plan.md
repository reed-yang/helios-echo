# Stage A Trainer/Dataloader Foundations Plan (design ch.2 D4-D7)

**Spec:** design doc ch.2 §2 (D4 eviction semantics, D5 write-source forward), §3 (D6 dataset unroll, D7 unroll loop + detach order), §1 (D1 config lineage, D3 per-stage recipe). Predecessors: transformer integration (D5's `return_final_hidden` requirement already landed as `capture_last_hidden`), trainer wiring (param groups, freeze channel, section-5 persistence), pipeline state machine.

**Goal:** the training-side substance that Stage A (and structurally Stage B) needs: dataset eviction slices, the section-level flow-loss primitive with correct backward/detach order, single-write simulation for Stage A, and the Stage A config fork. After this plan, `stage1_lora_mem368_A.yaml` is launchable on one node.

**Scope note:** D6/D7 are labeled Stage B in the design, but the loss primitive and dataset plumbing are one implementation unit; Stage A runs it at U=1 with `memory_single_write_prob≈0.5` mixing, Stage B only raises `memory_unroll_sections`. Stage C (D8/D9) stays out of scope.

## Task 1 — Dataset eviction slices (D4)

File: the stage1 dataset in `helios/utils/` (dataloader cited as `dataloader:` in the design; locate via `continue_source_latent` construction at :148-156 and `choice_idx` history slice at :188-191).

- New flag `return_evicted_latent: true` → dataset returns:
  - `evicted_latent = continue_source_latent[9(k-1) : 9k)` (9 latent frames — the oldest 9 of section k−1's history window);
  - its preceding 19-frame history slice (left zero-pad exactly 1 frame when k=3, where the slice start underflows);
  - `evicted_valid_frames = min(9, max(0, 9k − 19))` (int).
- Timeline invariant: `continue_source_latent = [19 zeros] + real frames`, so k=1 and k=2 evictions are all-zero (`evicted_valid_frames == 0` → trainer skips the write), k=3 is a partial write (1 zero + 8 real → frame-masked), k≥4 full. **Never** hardcode `choice_idx <= 1` boundaries — the valid-frame count is the single source of truth.
- Unit tests (CPU): synthetic latent tensor, assert slice indices and `evicted_valid_frames` for k=1,2,3,4; assert the k=3 left pad is exactly one frame of zeros.
- No changes to `tools/offload_data` (design D4: rejected offline write caching — write source is model hidden state and would go stale).

## Task 2 — Dataset unroll parameters (D6)

- Trainer passes `num_rollout_sections=memory_unroll_sections` into the dataset constructor (`train_helios.py:875-890` currently passes nothing; dataset default 3 gives 46 frames where U=4 needs 19+9U=55 — silent shape mismatch otherwise).
- Output dict gains `start_section_idx` and a per-section `prompt_embeds` list (reuse `_pick_prompt_embed(feature_data, start_section_idx+u)`, dataloader:282-303).
- `start_section_idx` sampling moves to the same seeded-generator discipline as `choice_idx` (dataloader:169-172) and is returned in the output dict (fixes divergence F3: global `random.randint` decoupled from the seeded per-sample stream).
- Index-time filter: drop samples with `total_sections < U`; the per-folder `dataset_cache.pkl` (dataloader:52-64) must be invalidated/rebuilt keyed on U.
- Unit test: `clean_all_latents` shape exactly `[B, 16, 19+9U, H, W]` for U∈{1,4}.
- **Pre-scale data budget gate:** metadata only guarantees `num_frame>=121` (dataloader:102-104) but U=4 needs ≥132 RGB frames; run an offline histogram scan of corpus section counts BEFORE budgeting Stage B (design: retention "cannot be verified from code"). Stage A (U=1) is unaffected.

## Task 3 — Section-level loss primitive + unroll loop (D7)

> Implementation decisions fixed after Task 1/2 landed (recorded pre-implementation):
> 1. **Per-sample write blending, batch-level write decision.** memory_single_write_prob
>    decides write-vs-static per BATCH; within a write batch, samples with
>    evicted_valid_frames==0 keep their prior state via a post-update blend
>    (save old query_state -> update() on the full batch -> torch.where(write_mask)
>    restore). valid==0 samples get a dummy all-True frame mask (their evicted
>    frames are zeros; the blend discards the result) so the module's
>    all-masked assert stays intact. No module changes.
> 2. **Token-level mask expansion.** frame_mask marks the LAST
>    evicted_valid_frames × tokens_per_frame tokens valid (zero-prefix frames
>    come first temporally); tokens_per_frame = E // 9 from the captured hidden.
> 3. **TF write sigma.** The Stage A/B write forward runs at timestep 0 on
>    clean evicted latents -> sigma_last = 0.0 for FiLM (inference-time writes
>    see sigma_last≈0.12-0.5; Stage C closes that gap per D9).
> 4. **Eviction resolution roles** (review blocker fix e064ad9): X_Noisy =
>    bucket-res evicted_latents; conditioning = full-res evicted_history_latents,
>    split into tiers with the SAME construction the trainer uses for the main
>    forward (evidence: logs/research/read-trainer-batch-prep-anchors.md).

File: `train_helios.py` + `helios/utils/utils_helios_base.py` (current `_flow_loss` backwards internally at :107).

- Refactor: extract a section-level primitive that **returns** the scalar loss (no internal backward); the outer `accelerator.accumulate` scope (train_helios.py:1484-1486) keeps ownership of sync boundaries.
- Unroll loop exactly per the design pseudocode: forward in time over U sections; per-section `accelerator.backward(loss_u / U)` under `no_sync` unless `sync_gradients and u == U-1` (conjunction with the outer accumulate state — ancestor config uses `gradient_accumulation_steps: 2`); rejected alternative (sum losses, single backward) OOMs by construction.
- **Detach order (the critical fix):** `state = state.detach()` at `(u+1) % memory_bptt_sections == 0`, placed AFTER this read's backward and BEFORE the next write — so `update(u)` stays in graph until `read(u+1)` consumes it. With bptt=1 the Enc/gate gradient comes exactly from the adjacent next-section read. Acceptance unit test: U=2, bptt=1 → Enc and gate gradients nonzero.
- Write forward: evicted 9 frames as X_Noisy, timestep=0, preceding-19 history, under `torch.no_grad()`, via the landed `capture_last_hidden` API; `mem.update(state, h, frame_mask(n_valid))` stays in graph (gradient enters Enc only through write→state→read→flow-loss). Skip when `evicted_valid_frames == 0`.
- Stage A mixing: with `memory_single_write_prob≈0.5`, alternate "pure static read (M₀)" and "single write then read" samples.
- DDP: keep `HELIOS_DDP_FIND_UNUSED` default true (write path is conditionally triggered), or zero-touch trick (utils_helios_base.py:100-105).
- Enc input note (D5): write-source sequence length varies by resolution bucket (full/half/quarter, dataloader:107-113) — `HeliosMemoryEncoder.update` already accepts variable-length sources; add a bucket-shape unit case to cover it.

## Task 4 — Assertion audit (D13 delta)

`validate_evolving_memory_config` already landed with the D13 core. Audit it against the full D13 list and add what's missing, most likely: `memory_tf_unroll ⇒ use_stage1_dataset and not use_stage3_dataset and memory_unroll_sections>=2 and not is_train_dmd and not use_error_recycling`; dataset-selection one-hot assertion; `validation_config.use_kv_cache ⇒ not is_enable_evolving_memory`. Unit tests per new assertion.

## Task 5 — Stage A config fork (D1/D3)

- `scripts/training/configs/stage1_lora_mem368_A.yaml` forked from `stage1_lora_cfr_368_correct.yaml` (inherits r128 LoRA, LR schedule, I2V-drop, corrupt/saturation recipe — yaml:44-51, 95-107, 142-171).
- Stage A settings: memory flags on, `memory_freeze_backbone: true` (only `evolving_memory.*` trains), memory LR `5e-5`, ~4k steps, U=1, `memory_single_write_prob: 0.5`.
- FORCE_LR guard: ancestor config depends on `HELIOS_FORCE_LR=1` (yaml:95-97, 105); the role-aware resume override landed in trainer wiring — assert it preserves the memory group LR in a unit test with a fake resume.
- Lineage completeness check per D13: enumerate lineage YAMLs, OmegaConf-merge each into the Args schema, assert memory keys explicitly present (`scripts/training/compare_yaml.py` A/B diff as manual aid only).

## Gates (mandatory order)

1. CPU unit suite green (Tasks 1-4 tests + existing 39).
2. Adversarial refute-review of the diff against real call sites (stewardship C) — before any GPU time.
3. 1-node GPU smoke profile: bs2, U=4, memory on — peak memory + s/it (design's bs4 single-section ~90 GiB anchor makes bs2×U=4 on H200 141 GB an UNTESTED assumption; this smoke revises the ch.2 §8 cost multipliers). MUST run multi-GPU (≥2) DDP and include steps where ranks hold mixed all-zero/valid eviction batches — the 3b review's blocking finding was a rank-desynchronized collective (fixed by rank-symmetric write participation: deterministic per-step coin + no local forward skip); the smoke proves the fix. find_unused_parameters was REFUTED as a failure mode (DDP traverses through the memory_tokens input edge; 2-rank repro confirmed gradients arrive).
4. Only after 1-3: Stage A launch decision (user-facing — cluster-scale training is not started autonomously).

## Execution routing

Task 1/2 (dataset) and Task 3 (trainer) are surgical edits on load-bearing files — main agent implements, with pre-read limited to the exact regions cited above. Task 4/5 are bounded and delegable (Sol high). Evidence-first: before editing, extract the current dataloader region (constructor args, slice code, cache logic) as an evidence card if it exceeds the direct-read budget.

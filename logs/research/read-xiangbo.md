# `mid_training_xiangbo` Branch Research Report

## Scope and evidence discipline

**OBSERVED branch state.** `/mnt/beegfs/siyuan/workspace/helios-team` is clean and checked out at `mid_training_xiangbo`, tracking `origin/mid_training_xiangbo`, with exactly five commits over `origin/main`: `0d2a5d3` (long-video evaluation suite), `9370669` (pre-`correct.yaml` working-state snapshot), `a613974` (author-recommended corrected Stage-1 recipe), `127d5bd` (lora368-correct monitor and `srun --mpi=none`), and `34f5a99` (training/eval/data/captioning sync). This was verified with `git status --short --branch`, `git log --stat origin/main..HEAD`, and `git rev-list --count origin/main..HEAD`.

Labels below are deliberate:

- **OBSERVED** means directly present in the branch code, configuration, or team documentation.
- **OBSERVED ABSENCE** means the requested files and their complete relevant implementations were inspected and the named capability is not present.
- **INFERRED / RECOMMENDED** means a design conclusion derived from the observations, not a claim that the branch already implements it.

## 1. What the mid-training effort is

### 1.1 The lora368 lineage and the actual object being trained

**OBSERVED.** The lora368 line is a Stage-1 flow-matching continuation of a Wan2.1/Helios-Base transformer, not a new architecture pretraining run and not Stage-2/Stage-3 distillation. The baseline `stage1_lora_cfr_368` loads `Helios-Base/transformer_init`, uses the 368×640 latent corpus, and resumes an existing adapter from checkpoint 9000; its launcher describes the 8-node, 64-H200, global-batch-128 continuation explicitly. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368.yaml:25-39`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368.yaml:87-104`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368.sbatch:3-20`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368.sbatch:29-35`]

**OBSERVED.** The trained parameter set is LoRA rank 128/alpha 128 on all linear layers plus `patch_embedding` LoRA, with the three Multi-Term Memory Conv3d modules (`patch_short/mid/long`) trained as full parameters; norms and the base 14B weights remain frozen. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:44-53`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:176-186`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:357-405`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:455-477`]

**OBSERVED.** Its loss is ordinary flow-matching MSE: the transformer predicts the target velocity; the loss is the weighted mean squared error between `model_pred` and `target`, followed by accelerator backward and gradient clipping. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:20-61`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:63-107`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:142-157`]

**OBSERVED.** The `_correct` run is a continuation of `stage1_lora_cfr_368/checkpoint-17000` into a new output directory, preserving the exact r128 adapter shape for state-dict compatibility. It uses the same ~388k merged CFR/B-prime latent corpus at 368×640, lowers LR from `3e-5` to `1e-5`, raises accumulation from 1 to 2, runs 40 GPUs (5 nodes × 8), reaches global batch 160, and targets step 22000 (about two additional epochs by the config’s accounting). [`/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_LORA368_CORRECT_CHANGES.md:10-16`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_LORA368_CORRECT_CHANGES.md:99-111`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:27-29`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:85-106`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:3-20`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:28-36`]

**OBSERVED.** The launcher symlinks only `checkpoint-17000` into the new output directory, uses environment-based DDP under Slurm, and sets `HELIOS_FORCE_LR=1` and `HELIOS_THROTTLE_FREE=1`; the monitor polls every five minutes, resubmits with a 25-minute floor, stops after three no-progress failures, and exits at step 22000. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:28-44`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:45-65`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/monitor_lora368c.sh:2-16`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/monitor_lora368c.sh:21-48`]

**OBSERVED configured descendants, not proof of completion.** The branch contains several follow-on recipes:

- `chunkcap`: starts from `_correct/checkpoint-21000`, trains 24,947 deterministically re-encoded clips with a per-chunk caption, and targets step 24000. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_chunkcap_368.yaml:26-29`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_chunkcap_368.yaml:85-105`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_chunkcap.sbatch:11-31`]
- `reweight`: starts from `_correct/checkpoint-19500`, trains a symlink-materialized multi-source mix of 1,396,442 weighted samples, global batch 120, through step 31140. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_reweight_368.yaml:27-44`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_reweight_368.yaml:100-119`]
- `rwtag`: an A/B from the same `_correct/checkpoint-19500`, but on a 154,015-sample tagged reweight-v2 mix, single-node global batch 120, through step 20784; caption embeddings are replaced from sidecars selected by `HELIOS_TAG_EMBED_DIR`. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_rwtag_368.yaml:27-41`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_rwtag_368.yaml:97-116`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_rwtag.sbatch:11-20`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_rwtag.sbatch:38-43`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_rwtag.sbatch:61-72`]
- `neworg`: starts from `reweight/checkpoint-27500`, runs 1000 steps on the canonical natural-count 12-source tree, and changes the run shape to one node/global batch 24. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_neworg_368.yaml:26-43`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_neworg_368.yaml:99-115`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_neworg.sbatch:11-17`]

### 1.2 `correct.yaml` versus the released Stage-1 configs

**OBSERVED.** `correct.yaml` is a 28-line patch overlay, not a complete standalone training config. It adds/overrides dynamic-shifting-at-validation, I2V-format sampling, random history corruption (Noise + Blur), and saturation augmentation. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/correct.yaml:1-27`]

**OBSERVED.** The signed-in released Stage-1 init/post configs are otherwise nearly identical: init starts from raw `Wan-AI/Wan2.1-T2V-14B-Diffusers` at LR `5e-5`; post starts from `BestWishYsh/Helios-Base/transformer_init` at LR `3e-5`; both use r128 all-linear LoRA, full memory patches, noise-only history corruption, no I2V-drop field, and no enabled saturation field. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage_1_init.yaml:31-59`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage_1_init.yaml:91-100`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage_1_init.yaml:144-176`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage_1_post.yaml:31-60`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage_1_post.yaml:92-101`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage_1_post.yaml:145-177`]

**OBSERVED.** Team documentation records the author communication behind this patch: Base/Mid were trained without fully enabling Easy Anti-Drifting and without ever training the I2V context format; the upstream correction was supplied in response to issue #38. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_CORRECT_RECIPE.md:1-7`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_CORRECT_RECIPE.md:11-31`]

### 1.3 Why each correction exists and how it operates

#### A. Noise-only → random Noise/Blur/clean history corruption

**OBSERVED.** The baseline uses `corrupt_mode_history: "noise"`, so its configured downsample ratios cannot be reached; `_correct` changes the mode to `"random"` and probability to `0.8889`. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368.yaml:153-163`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:156-166`]

**OBSERVED mechanism.** `corrupt_history_latents` first returns clean history with probability 0.1. Otherwise, random mode chooses noise with probability 0.8889 and downsample with the remainder, giving approximately 0.8 noise / 0.1 blur / 0.1 clean overall. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:283-318`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:156-166`]

**OBSERVED mechanism.** Blur is bilinear downsample-then-upsample with antialiasing; for 5D video latents it reshapes `(B,C,T,H,W)` to frame batches, processes spatial dimensions, then restores the original layout. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:188-213`]

**INFERRED purpose, consistent with team documentation.** This trains the model against spatial low-pass damage resembling autoregressive history degradation, rather than only Gaussian corruption. The team explicitly calls Blur a missing anti-drift component and states that noise-only made the downsample settings dead configuration. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_LORA368_CORRECT_CHANGES.md:28-42`]

#### B. I2V-format training

**OBSERVED.** `_correct` adds `random_drop_i2v_ratio: 0.1`; the baseline does not set it, and its dataclass default is zero. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:142-145`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/train_config.py:229-236`]

**OBSERVED mechanism.** T2V drop is tested first. Only in its `else` branch can I2V drop fire; I2V then zeros `hist_seq_len - 1` history frames from oldest to newest, preserving the first-frame anchor and newest history frame when the short stream is assembled. At T2V probability 0.4 and conditional I2V probability 0.1, the effective I2V frequency is `(1-0.4)×0.1≈6%`. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:655-690`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:691-730`]

**INFERRED purpose, consistent with the upstream overlay.** This closes a training/inference format mismatch behind slow I2V startup: the overlay says inference had relied on zero-shot behavior because training never constructed “first-frame anchor + last-frame” history. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/correct.yaml:8-15`]

#### C. Saturation augmentation

**OBSERVED.** `_correct` enables `is_add_saturation`, uses a 0.1 clean probability and factor range 0.3–1.7. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:167-171`]

**OBSERVED mechanism.** The function computes a per-pixel mean over latent channels and scales deviations from that mean by a factor sampled either below 1 or above 1; the x0 anchor branch is deliberately not altered, while nonzero history windows independently skip with `saturation_clean_prob` or receive the transform. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:487-508`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:510-555`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:556-620`]

**OBSERVED correction to the code path.** Before this branch patch, saturation was wired only into Stage-3/post paths; the branch now invokes it in `prepare_stage1_noise_input` after history corruption and before target corruption. Merely setting the YAML flag would otherwise have been a Stage-1 no-op. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:826-863`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_CORRECT_RECIPE.md:89-126`]

**INFERRED purpose, consistent with team documentation.** This makes the denoiser robust to recursive color/chroma shifts and targets long-video fading/overexposure. The evidence cited by the team came from Distilled/Stage-3 ablation, so applying it in Stage-1 is explicitly an extrapolation rather than a Stage-1-controlled result. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_CORRECT_RECIPE.md:137-142`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_LORA368_CORRECT_CHANGES.md:54-75`]

#### D. Resume-time learning-rate correction

**OBSERVED.** `accelerator.load_state` restores optimizer and scheduler state; under `HELIOS_FORCE_LR=1`, the branch rewrites optimizer `lr`/`initial_lr` and scheduler `base_lrs` from the current config after loading. Without this, the intended `3e-5→1e-5` continuation could silently retain the old checkpoint LR. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:1059-1085`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:1086-1104`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:53-56`]

**OBSERVED validation.** The team records a smoke run in which Blur, I2V-drop, and Saturation debug branches all fired, resume/LR override worked, and the loss remained finite; the short 200-clip smoke later hangs from cross-rank tail alignment, which is documented as a small-data artifact. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_LORA368_CORRECT_CHANGES.md:115-120`]

## 2. Drift evidence measured by the team

### 2.1 VS24 long-video checkpoint evaluation

**OBSERVED protocol.** The team evaluated (a) seven checkpoints × 20 single-prompt videos at ~91 seconds/2178 frames and (b) fourteen checkpoints × 40 prompt-switching videos at ~58 seconds/1386 frames with about six prompts per video. `step-0` is the pre-continuation base. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_ANALYSIS.md:1-16`]

**OBSERVED completeness.** The long set completed DOVER, PickScore, HPSv3, and all seven VBench dimensions on full/start15/end15. The event-switch set completed the three quality/preference metrics but only partial VBench; HeliosBench did not run for either set in that campaign. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_ANALYSIS.md:19-31`]

**OBSERVED main effect at ~91 s.** `step-0` was best on per-frame quality/preference but nearly static: dynamic degree was 0.30 versus 0.55–0.90 for trained checkpoints. From step 0 to trained models, DOVER overall dropped by roughly 0.10, imaging quality by about 10 points, and HPSv3 by about 4–5, while motion approximately doubled or tripled. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_ANALYSIS.md:35-59`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_REPORT.md:27-46`]

**INFERRED interpretation adopted by the team.** This is a motion-versus-fidelity tradeoff, not a simple “training made everything worse”: the visually high-scoring base partly wins by barely moving. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_ANALYSIS.md:53-59`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_ANALYSIS.md:150-154`]

**OBSERVED temporal stability at ~91 s.** Drift is `|end15-start15|`. Among trained checkpoints, step 5000 had the lowest DOVER-overall drift (0.069) and HPSv3-mean drift (0.862), while step 0 retained the lowest PickScore-mean drift (0.417); the team selected step 5000 as the best-rounded trained checkpoint because it combined relatively low drift with dynamic degree 0.75 and mid-pack quality. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_ANALYSIS.md:61-71`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_REPORT.md:48-69`]

**OBSERVED event-switch behavior at ~58 s.** The same early quality drop appears, but trained checkpoints are largely flat/noisy rather than monotonic. HPSv3 start/end drift is about 2.0–3.2, materially larger than the long set’s ~0.9–1.5. The report warns that PickScore/HPSv3/semantic score all frames against only the first prompt, so prompt changes confound “drift”; no firm best event-switch checkpoint is justified without per-event prompt-aware scoring. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_ANALYSIS.md:73-99`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_REPORT.md:11-16`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_REPORT.md:77-123`]

### 2.2 Full-FT stability evidence

**OBSERVED failed variant.** `stage1_fullft_cfr` unfreezes the complete 14.31B DiT while keeping VAE and text encoder frozen. It was stopped around step 2400 because samples showed deformation and temporal jitter. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLSFT_ANALYSIS.md:1-6`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLSFT_ANALYSIS.md:9-47`]

**OBSERVED training signature.** Over steps 0–2468, median loss stayed around 0.072, median grad norm was 0.086, and the maximum grad spike was 4.95; after a 200-step warmup, LR remained constant at `1e-5`. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLSFT_ANALYSIS.md:50-69`]

**INFERRED root cause recorded by the team.** Flat loss plus visibly degrading samples was interpreted as catastrophic drift/forgetting rather than objective improvement. The proposed causal combination was full-rank movement including norm/AdaLN and time/structural layers, LR `1e-5`, no EMA, global batch 32, no LR decay, strong history corruption, and off-distribution terse captions; temporal-prior drift manifested as jitter and spatial-prior drift as deformation. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLSFT_ANALYSIS.md:71-85`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLSFT_ANALYSIS.md:89-120`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLSFT_ANALYSIS.md:129-134`]

**OBSERVED comparison variants.** LoRA r128 remained stable through step 9000, while the Round-2 hybrid SFT (r256 LoRA + full patch embedding + full memory patches, base and norm/AdaLN frozen) completed 6000 steps. The comparison attributes Round-2 stability to LR `2e-6`, EMA, global batch 128, reduced trainable scope, and softer corruption. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLFT_COMPARISON_AND_STABILITY.md:6-15`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLFT_COMPARISON_AND_STABILITY.md:68-79`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLFT_COMPARISON_AND_STABILITY.md:85-116`]

**OBSERVED operational conclusion.** The team recommends, in priority order, lower and decayed LR, EMA evaluation, effective batch roughly 128–512, frozen norm/AdaLN and cautious time embedding, then weaker augmentation; fallback to LoRA/hybrid SFT is explicitly recommended if full-rank movement is unnecessary. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLFT_COMPARISON_AND_STABILITY.md:142-156`]

**OBSERVED unresolved experiment.** Round-3/fullbase uses a selectively merged LoRA-17000 start, full-trains the base linear backbone and memory patches, preserves patch embedding as LoRA, freezes norm/AdaLN, uses LR `1e-6`, batch 128, and no EMA because the existing EMA helper is LoRA-specific. Its documented result is still “pending evaluation”; it must not be cited as evidence that this scope is stable. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLFT_COMPARISON_AND_STABILITY.md:180-203`]

## 3. Wan2.2 port status and target choice

**OBSERVED: underway and smoke-validated, not abandoned.** `WAN22_PORT.md` describes a concrete port from Wan2.1-T2V-14B to dense Wan2.2-TI2V-5B, including Stage-1 init/post configs, VAE re-encoding, merge, inference, and evaluation scripts. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/WAN22_PORT.md:1-5`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/WAN22_PORT.md:44-53`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/WAN22_PORT.md:63-96`]

**OBSERVED implementation delta.** Wan2.2 changes latent channels 16→48, spatial VAE compression 8×→16×, inner dimension 5120→3072, and transformer layers 40→30 while preserving 4× temporal compression and 33-RGB-frame chunk geometry. The only model-code hardcode changed is initialization of memory-patch input channels: it now slices to `self.patch_short.in_channels` rather than 16. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/WAN22_PORT.md:6-29`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1087-1109`]

**OBSERVED resolution constraint.** The port uses 384×640, not 368×640, because 368/16 produces odd latent height 23 and stride-2 patch embedding would drop a row; the Wan2.2 config records 384/16=24 and a distinct 384×640 latent directory. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/WAN22_PORT.md:31-42`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_init_wan22.yaml:12-29`]

**OBSERVED validation status.** The document reports a clean 5.04B load with only six newly initialized multi-term-patch keys, 48-channel encoded latents, and a 30-step six-GPU smoke train with finite loss, checkpoint save, and validation video; therefore both training and inference paths were exercised. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/WAN22_PORT.md:98-103`]

**OBSERVED ABSENCE: no target-policy decision.** `WAN22_PORT.md` does **not** say that Wan2.2-5B replaces Wan2.1-14B for new work, nor does it recommend abandoning the 14B line; it is a technical port/runbook and status report. The active lora368 lineage still explicitly targets Helios-Base/Wan2.1-era `transformer_init`. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/WAN22_PORT.md:1-103`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:32-48`]

**INFERRED decision for downstream design.** Treat Wan2.1-14B as the branch’s measured mid-training target and Wan2.2-5B as a validated secondary port. Do not select 5B merely because `WAN22_PORT.md` exists; the document provides feasibility evidence, not a project-priority mandate. Chunk-aligned captioning itself is designed to be common to both because both use 4× temporal compression and 33-frame chunks. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/CHUNK_ALIGNED_CAPTIONING.md:20-31`]

## 4. `long_video_eval`: exact coverage, missing time-series metrics, and extension path

### 4.1 What the suite itself does

**OBSERVED.** `long_video_eval` is primarily a config/resource/orchestration layer. It parses JSON configuration and CLI overrides, validates input/resource availability, builds a prompt/video manifest, optionally stages arbitrary video roots into a normalized symlink tree, and executes enabled `external_command` steps. It does not implement the substantive metrics in the package itself. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/cli.py:16-39`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/cli.py:42-88`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/runner.py:16-35`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/runner.py:55-107`]

**OBSERVED.** The input manifest contains one prompt string per indexed video/model, duration frames, models, GPUs, and paths. It has no event-boundary array or per-event prompt schema. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/manifest.py:12-18`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/manifest.py:41-99`]

### 4.2 Exact metrics

**OBSERVED HeliosBench backend.** Seven enabled metrics are supported:

1. `aesthetic`
2. `motion_amplitude`
3. `motion_smoothness`
4. `semantic`
5. `drifting_aesthetic`
6. `drifting_motion_smoothness`
7. `drifting_semantic`

The helper maps these to the official evaluation scripts and identifies each aggregate/per-video result key. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/heliosbench_on_our_prompts_helper.py:17-60`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/heliosbench_on_our_prompts_helper.py:224-298`]

**OBSERVED segment backend.** Videos are evaluated as `full`, `start15` (first 15%), and `end15` (last 15%); the production backend writes frame-accurate start/end clips using `int(total_frames*ratio)`. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/prepare_helios_ratio_segments.py:58-87`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/prepare_helios_ratio_segments.py:90-145`]

The regular metrics are:

- **DOVER:** normalized aesthetic, technical, overall, plus aesthetic/technical raw scores. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/metrics/dover_metric.py:39-50`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/metrics/dover_metric.py:52-90`]
- **PickScore:** sampled-frame mean, minimum, and top-30%-mean prompt preference. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_segment_metric.py:289-331`]
- **HPSv3:** sampled-frame mean, minimum, and top-30%-mean human-preference reward. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_segment_metric.py:263-286`]
- **VBench custom-video:** subject consistency, background consistency, temporal flickering, motion smoothness, dynamic degree, aesthetic quality, and imaging quality. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_ratio_eval_sharded_visko3.sh:46-58`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_ratio_eval_sharded_visko3.sh:303-354`]

**OBSERVED aggregation.** For every scalar metric, the summary reports full mean, start mean, end mean, signed `end-start` mean, absolute drift mean, and sample counts. VBench is aggregated in the same form. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/summarize_helios_long_ratio_eval.py:68-112`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/summarize_helios_long_ratio_eval.py:152-180`]

**OBSERVED frame sampling.** PickScore/HPSv3 default to one sampled frame per 33-frame Helios chunk in the long suite; the sampler chooses the chunk midpoint for one frame or evenly spaced offsets for denser sampling. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_ratio_eval_sharded_visko3.sh:39-44`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_segment_metric.py:86-129`]

**OBSERVED VideoAlign status.** A VideoAlign metric adapter exists and emits VQ/MQ/TA/Overall, but the default 60s/90s suite intentionally excludes it because its long-video sampling and memory/context validity are unresolved. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/metrics/videoalign_metric.py:13-49`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/README.zh.md:53-58`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/QUICKSTART.zh.md:192-201`]

### 4.3 Durations and event-switch configuration

**OBSERVED.** The generic/local and parity configs ship a 60-second-class setup at 1452 frames and 24 fps; the concrete VS24 long config is 2178 frames (~90.75 s), while the event-switch config is 1386 frames (~57.75 s). Despite the generic filename `helios_60s90s`, there is no separate shipped generic 90-second config; `vs24_long.json` is the concrete ~90-second configuration inspected here. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/configs/helios_60s90s.local.example.json:2-11`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/configs/helios_60s_long100_original_base.pipeline_parity.json:2-11`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/configs/vs24_long.json:2-17`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/configs/vs24_eventswitch.json:2-24`]

**OBSERVED.** Runs are entirely config-selected: each config names model/checkpoint directories, GPU IDs, duration, metric list, ratio, VBench flag, and frame-sampling density. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/configs/vs24_long.json:118-170`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/configs/vs24_eventswitch.json:47-92`]

**OBSERVED limitation.** The event-switch config points to one ordinary `prompt.txt`, and the manifest stores one prompt per video. Consequently, current text-alignment metrics cannot score each event against its own prompt. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/configs/vs24_eventswitch.json:2-8`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/manifest.py:12-18`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/VS24_EVAL_ANALYSIS.md:88-99`]

### 4.4 Requested slope/curve/boundary metrics: what is absent

**OBSERVED ABSENCE after reading the complete suite package, all four shipped configs, and the backend scripts named above.** There is no within-video time-series result schema and no regression over temporal windows. The report’s “trend” is only checkpoint-first-versus-checkpoint-last and best-checkpoint selection, not a video-time slope. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/make_eval_report.py:91-111`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/make_eval_report.py:129-157`]

Specific absences:

- **No motion-amplitude-over-time curve or slope.** There is one whole-video HeliosBench `motion_amplitude` and segment-level VBench `dynamic_degree`; the only motion drift metric is `drifting_motion_smoothness`, not motion-amplitude slope. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/heliosbench_on_our_prompts_helper.py:24-35`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/heliosbench_on_our_prompts_helper.py:42-58`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_ratio_eval_sharded_visko3.sh:55-57`]
- **No saturation/colorfulness curve or slope.** Saturation exists only as a training augmentation in the files reviewed above; no evaluation backend emits saturation/chroma statistics. The exhaustive regular summary accepts whatever numeric metric keys appear, but existing producers emit only DOVER/PickScore/HPSv3/VideoAlign and VBench results. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_segment_metric.py:31-57`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_segment_metric.py:377-403`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/summarize_helios_long_ratio_eval.py:43-49`]
- **No boundary LPIPS.** No suite config or metric dispatcher contains LPIPS or boundary-pair handling; the dispatcher choices are only DOVER, PickScore, HPSv3, and VideoAlign, plus externally launched VBench. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_segment_metric.py:31-57`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_segment_metric.py:377-403`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_ratio_eval_sharded_visko3.sh:20-27`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_ratio_eval_sharded_visko3.sh:46-58`]

### 4.5 How to add the missing signals

**RECOMMENDED minimal integration.** Add a new script such as `scripts/evaluation/run_helios_long_timeseries_metric.py` and register it as another `external_command` step in the JSON config. This requires no change to the suite core because its runner renders arbitrary environment/command entries and checks subprocess success. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/runner.py:55-82`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/runner.py:85-107`]

Recommended output contract per video:

```text
{
  video_id, model,
  windows: [{chunk_idx, t_norm, motion_amp, saturation, colorfulness}],
  event_boundaries: [{left_chunk, right_chunk, boundary_lpips, control_lpips}],
  summaries: {motion_slope, saturation_slope, saturation_auc,
              boundary_lpips_mean, boundary_excess_over_control}
}
```

**RECOMMENDED computation.** Use 33-frame-aligned windows, matching the existing chunk sampler. Compute robust optical-flow magnitude per window for `motion_amp`; compute RGB/HSV saturation and preferably CIELAB chroma per window for the color curve; fit slope against normalized time and retain the raw series, not just the slope. For boundary LPIPS, compare the last frame before and first frame after each chunk/event boundary, then subtract or ratio-normalize against matched adjacent-frame pairs inside chunks so ordinary motion is not mislabeled as a seam. The existing helper already provides deterministic 33-frame sampling primitives that can be reused. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/evaluation/run_helios_long_segment_metric.py:86-129`]

**RECOMMENDED event-aware extension.** Add an optional `event_spec_file` to defaults and include per-event prompts and switch frames in the generated manifest. Without that addition, the suite cannot distinguish an intended semantic event switch from drift because its `VideoItem` has only one prompt. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/manifest.py:12-18`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/long_video_eval/long_video_eval/config.py:39-53`]

## 5. Training-operations constraints and full-FT versus LoRA conclusions

### 5.1 Cluster and run shape

**OBSERVED.** The team operates on Slurm H200 nodes with datasets/checkpoints under `/mnt/beegfs`; the principal `_correct` run is five nodes × eight H200s, one `srun` task per GPU, bf16, per-GPU batch 2, accumulation 2, global batch 160. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:1-10`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:11-20`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage1_lora_cfr_368_correct.yaml:75-106`]

**OBSERVED.** `mc-node02` lacks the PMIx plugin needed by the default `srun` MPI mode, so the corrected launchers use `srun --mpi=none`. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:39-44`]

**OBSERVED.** A recurring node constraint is that c-node07 works for single-node NVLink training but wedges on its first cross-node NCCL collective, so it is excluded from current five-node jobs and used for the single-node `rwtag` run. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_reweight.sbatch:15-18`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_rwtag.sbatch:18-20`]

**OBSERVED.** `skip_dataloader_dcp` exists because multi-node `torch.distributed.checkpoint` dataloader-state saving can crash with NCCL Error 2. Skipping it preserves model/optimizer state but loses exact mid-epoch iterator position and permits LoRA resume with a changed rank count. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/train_config.py:147-160`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:2134-2145`]

**OBSERVED.** The branch gates repeated hot-loop `gc.collect()+empty_cache` behavior behind `HELIOS_THROTTLE_FREE`; the launcher records a validated ~26% step-time reduction and instructs removing the export if OOM appears. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:159-180`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:1386-1396`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:18-19`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_lora_cfr_368_correct.sbatch:55-56`]

### 5.2 Checkpoint and partial-merge tooling

**OBSERVED.** A normal Stage-1 LoRA checkpoint needs both `pytorch_lora_weights.safetensors` and `transformer_partial.pth`, because the memory patch Conv3d modules are full-trained rather than LoRA-trained. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_STAGE1_INIT_VS_POST.md:97-107`]

**OBSERVED.** The requested `merge_lora_partial_for_helios.py` is **selective**, not a generic “merge everything” utility. It merges all linear LoRA deltas into the base, overwrites `patch_short/mid/long` from the partial checkpoint, intentionally leaves base `patch_embedding` unmerged, and emits its LoRA tensors separately for the hybrid fullbase run. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/merge_lora_partial_for_helios.py:1-24`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/merge_lora_partial_for_helios.py:78-124`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/merge_lora_partial_for_helios.py:126-151`]

**OBSERVED.** The merge job is CPU-only, requests 256 GB RAM, loads the 14B base in fp32 (~57 GB per its launcher comment), and produces the LoRA-17000 start consumed by the fullbase recipe. [`/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/merge_lora368_partial.sbatch:1-13`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/merge_lora368_partial.sbatch:21-30`]

### 5.3 LoRA versus full-FT operational conclusion

**OBSERVED.** LoRA uses plain DDP because each H200 can hold the frozen 14B bf16 base plus adapter optimizer state; full-FT requires DeepSpeed ZeRO-2. The full-FT launcher uses four nodes/32 H200s and is world-size locked for resume; the fullbase hybrid uses two nodes/16 H200s, batch 128, and stages the 27 GB merged transformer into node-local `/dev/shm` to avoid many-reader BeeGFS SIGBUS failures. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/HELIOS_STAGE1_TRAINING.md:24-44`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_fullft_cfr.sbatch:3-15`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_fullbase_cfr368.sbatch:3-17`, `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/train_stage1_fullbase_cfr368.sbatch:61-81`]

**INFERRED decision from measured evidence.** LoRA is the established stable default for memory-mechanism prototyping. Full-FT has a documented failure at 2400 steps, the safer fullbase scope has not yet reported quality results, and full-FT introduces ZeRO/world-size/checkpoint/EMA complexity. A new learnable memory module can be full-trained while retaining LoRA on the backbone, matching the already-safe pattern used for `patch_short/mid/long`, unless an ablation proves full-backbone adaptation is required. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLFT_COMPARISON_AND_STABILITY.md:68-79`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLFT_COMPARISON_AND_STABILITY.md:142-156`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/FULLFT_COMPARISON_AND_STABILITY.md:180-203`]

## 6. Functional summary of branch `train_helios.py` changes

Relative to `origin/main`, the branch changes training behavior in these functional groups:

1. **DDP unused-parameter performance gate.** `HELIOS_DDP_FIND_UNUSED=0` disables the extra DDP graph traversal; `_flow_loss` adds a zero-valued touch over all trainable parameters so conditionally unused history parameters still receive zero gradients. Default behavior remains `find_unused=True`. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:146-152`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:95-105`]

2. **True full-FT and a hybrid scope.** `is_full_finetune` either unfreezes the whole transformer without adding LoRA or selects `base_linear_lora_patch`, which full-trains the backbone and memory patches while freezing norm/AdaLN and base patch embedding and attaching only a patch-embedding LoRA. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:397-456`]

3. **Patch-embedding LoRA continuation.** The hybrid scope can seed that adapter from a separate safetensors file produced by the selective merge tool, rather than Gaussian-initializing it. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:416-450`]

4. **Scope auditing and configuration guards.** It counts and asserts that patch-embedding base and norm/AdaLN have zero trainable parameters while patch LoRA and memory patches are trainable; startup guards restrict the accepted scope names and reject incompatible generic patch-training flags. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:479-507`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:2687-2700`]

5. **Full-FT save/load semantics.** For full-FT, the pre-save hook defers full model/optimizer storage to the DeepSpeed engine and avoids a rank-0 28 GB save inside the collective hook; the load hook restores through accelerator/DeepSpeed and optionally restores dataloader DCP. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:648-704`]

6. **Memory avoidance for full-FT.** The branch does not upcast the entire 14B transformer to fp32 before optimizer setup because DeepSpeed maintains its own fp32 master copy. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:805-818`]

7. **Resume LR authority.** `HELIOS_FORCE_LR` reapplies config LR to optimizer and scheduler after checkpoint restore. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:1059-1104`]

8. **Hot-loop cache-thrash gates.** Several `free_memory()` calls are skipped when `HELIOS_THROTTLE_FREE=1`. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:1386-1396`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:2009-2015`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:2379-2405`]

9. **Optional dataloader-state checkpointing.** `skip_dataloader_dcp` bypasses only the dataloader DCP save while leaving `accelerator.save_state` model/optimizer checkpointing intact. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:2134-2145`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/train_config.py:151-160`]

10. **Final full-model export.** At final save, full-FT writes a standalone HF transformer; the hybrid first fuses and unloads patch-embedding LoRA. The ordinary path still writes LoRA plus extra components. [`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:2411-2464`]

**OBSERVED adjacent branch changes outside `train_helios.py`.** The Stage-1 dataloader now supports random caption-version selection, event-to-chunk prompt lookup, tagged embedding sidecars, and incremental text sidecars. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/dataset/dataloader_history_latents_dist.py:35-37`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/dataset/dataloader_history_latents_dist.py:201-280`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/dataset/dataloader_history_latents_dist.py:282-303`]

## 7. Multi-event and chunk-aligned-captioning constraints for an evolving memory mechanism

### 7.1 Existing contracts

**OBSERVED chunk contract.** One bidirectional model chunk is 9 latent frames = 33 RGB frames. Chunk-aligned captions cover non-overlapping `[cut_start+33k, cut_start+33(k+1))` intervals; tails shorter than 33 frames are discarded, and boundaries must be frame-based rather than seconds because clip FPS varies. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/CHUNK_ALIGNED_CAPTIONING.md:1-24`]

**OBSERVED independent VAE encoding.** Multi-event latents encode each 33-frame chunk independently, then stack them; there is no VAE latent leakage across chunk boundaries. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/offload_data/get_multievent-latents.py:220-230`]

**OBSERVED training prompt binding.** The offline encoder embeds every event caption, assigns each chunk by maximum RGB-frame overlap (ties go to the earlier event), and stores `event_prompt_embeds` plus `event_idx_per_chunk`. The dataloader samples one target chunk and returns the prompt of the event owning that chunk; transformer forward and loss remain otherwise unchanged. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/offload_data/get_multievent-latents.py:56-97`, `/mnt/beegfs/siyuan/workspace/helios-team/tools/offload_data/get_multievent-latents.py:233-270`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/dataset/dataloader_history_latents_dist.py:282-303`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/dataset/dataloader_history_latents_dist.py:345-379`]

**OBSERVED inference binding.** Multi-event inference uses the same majority-overlap rule, converts event ownership to per-event chunk counts, drops events that own zero chunks, and hard-switches prompts at chunk boundaries with `interpolation_steps=0`. [`/mnt/beegfs/siyuan/workspace/helios-team/infer_multievent.py:60-89`, `/mnt/beegfs/siyuan/workspace/helios-team/infer_multievent.py:156-186`]

**OBSERVED old-data incompatibility.** Existing old latents may contain an unrecorded random temporal crop offset and length buckets not divisible by 33. Exact chunk-caption supervision therefore requires deterministic re-encoding from `cut_start` for exactly `n_chunks×33` frames; old latents cannot safely receive the new per-chunk labels. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/CHUNK_ALIGNED_CAPTIONING.md:69-83`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/CHUNK_ALIGNED_CAPTIONING.md:149-174`]

**OBSERVED loader hazard.** Although the captioning document requires assertions that map length equals chunk count and indices are in range, the current loader clamps `choice_idx` to the last map entry rather than asserting exact length. A malformed short map can therefore silently reuse its final event assignment. [`/mnt/beegfs/siyuan/workspace/helios-team/docs/CHUNK_ALIGNED_CAPTIONING.md:33-47`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/dataset/dataloader_history_latents_dist.py:291-298`]

### 7.2 Implications for Echo-style evolving memory

**INFERRED / REQUIRED FOR COMPATIBILITY.** Memory updates must be chunk-synchronous. A write/update should occur after a complete 9-latent/33-RGB-frame chunk, because both supervision and hard prompt switches are defined at that boundary; per-frame writes would not align with the current training contract.

**INFERRED / REQUIRED FOR COMPATIBILITY.** A prompt switch must not reset visual memory. The new prompt conditions the current target chunk, while the latent history remains the previous visual sequence. Therefore memory state should persist across event IDs, with any reset controlled by sequence/video boundaries rather than text boundaries.

**INFERRED training gap.** Current Stage-1 sampling chooses one random target chunk per clip and reconstructs only its fixed 19-frame latent history; it does not execute preceding chunks sequentially to evolve a learned state. A truly evolving Echo memory cannot learn its recurrent write dynamics from this path alone; it needs an unrolled multi-chunk mode (or precomputed teacher memory states) while retaining the existing one-chunk flow loss as a warm-start path. [`/mnt/beegfs/siyuan/workspace/helios-team/helios/dataset/dataloader_history_latents_dist.py:136-196`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:1343-1373`]

**INFERRED positional contract.** Event boundaries can fall inside a 33-frame chunk; the current rule assigns the whole mixed chunk to the event with majority overlap. Memory writes must still consume the actual generated mixed visual chunk rather than an idealized event-only representation, or train/inference semantics will diverge. [`/mnt/beegfs/siyuan/workspace/helios-team/tools/offload_data/get_multievent-latents.py:61-87`, `/mnt/beegfs/siyuan/workspace/helios-team/docs/CHUNK_ALIGNED_CAPTIONING.md:73-82`]

**INFERRED evaluation requirement.** A memory mechanism should be judged on within-video slopes and event-boundary behavior, not only endpoint means. At minimum, add motion-amplitude slope, saturation/chroma slope, and boundary-excess LPIPS, plus per-event prompt alignment. Current start/end metrics can miss mid-clip collapse/recovery and currently confound intended prompt changes with semantic drift.

## TL;DR

- `mid_training_xiangbo` is clean and five commits ahead of `origin/main`; the core effort is Stage-1 Wan2.1/Helios-Base LoRA continuation, not Stage-2/3 distillation.
- lora368-correct resumes r128 checkpoint 17000 on the same ~388k 368×640 latents, uses 40 H200s/global batch 160, LR `1e-5`, and targets step 22000.
- The corrected recipe adds three missing input-side robustness modes: ~10% Blur, ~6% effective I2V-format samples, and latent saturation factors 0.3–1.7; Stage-1 saturation required a code-path patch.
- `HELIOS_FORCE_LR` is essential because resume restores the old optimizer/scheduler LR; `HELIOS_THROTTLE_FREE` and skipped dataloader DCP address measured cluster performance/reliability problems.
- VS24 shows a real motion–fidelity tradeoff: continuation raises dynamic degree from 0.30 to 0.55–0.90 but lowers per-frame quality; step 5000 was the best-rounded trained checkpoint at ~91 s.
- Prompt-switch evaluation is not event-aware: text metrics use only the first prompt, so its higher start/end “drift” cannot rank event adherence reliably.
- Full 14.31B FT failed around step 2400 with deformation/jitter, flat loss, and grad spikes; LoRA remained stable, while the safer fullbase experiment still lacks reported quality results.
- Wan2.2-TI2V-5B is a real, smoke-validated port, not abandoned, but no document declares it the replacement target; the measured mid-training line remains Wan2.1-14B.
- The eval suite computes HeliosBench, DOVER, PickScore, HPSv3, and seven VBench dimensions over full/start15/end15, but has no within-video slope, saturation curve, or boundary LPIPS.
- Add missing time-series metrics as a new config-driven external backend and extend the manifest with per-event prompts/boundaries.
- Multi-event/chunk-caption contracts are exactly 33 RGB frames per chunk with hard prompt switches; evolving memory must persist across prompt switches and update at chunk boundaries.
- Current Stage-1 samples one target chunk with fixed latent history, so an Echo-style learned recurrent memory needs an additional multi-chunk unrolled training path.

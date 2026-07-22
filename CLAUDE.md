# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## This working copy: helios-echo

`helios-echo/` is a **working fork of `../helios-team` @ `mid_training_xiangbo`** dedicated to porting Echo-Infinity's learnable evolving memory into Helios. Work happens on branch **`echo-memory`**; branch `mid_training_xiangbo` tracks `team/mid_training_xiangbo` (remote `team` = local `../helios-team`, remote `origin` = `github.com/Visko-Platform/helios-team.git`). Sync upstream with `git fetch team && git rebase mid_training_xiangbo` (or merge) after updating the sibling checkout.

Read order for this project:
1. `docs/echo-to-helios-migration-design.md` — master design (overview + 4 verified chapters). Start at 总纲 §一 (unified baseline) and §二 (prior-vs-code divergence table).
2. `logs/findings.md` — key findings, engineering landmines, fact corrections.
3. `logs/progress.md` — current status and next steps (K0 checklist).
4. `logs/research/read-*.md` — 6 deep-read reports (Echo + Helios internals, all file:line anchored).

The sections below this one document the **inherited helios-team codebase** and remain valid for this tree.

## What this is

`helios-team/` is a full clone of the **upstream public Helios repo** (`github.com/PKU-YuanGroup/Helios`) — Helios is a Wan2.1-T2V-14B model re-architected into an **autoregressive, chunked video DiT** for real-time long-video generation (claimed 19.5 FPS on a single H100). It is a *sibling* of the `helios/` tree referenced by the parent workspace `../CLAUDE.md`; that parent doc describes a different, locally-modified checkout (single 2600-line trainer, Stage-4 multi-event SFT). **This tree has no Stage-4 / multi-event code** — it is the released Base/Mid/Distilled three-stage pipeline only. Don't assume the parent doc's `file:line` pointers apply here.

There is no test suite, no CI, and no build step. The repo is driven entirely through `accelerate launch` (training) and `python infer_helios.py` (inference), both wrapped in `scripts/`.

## Setup / run

```bash
bash install.sh                 # pip install -r requirements.txt, then PIN triton==3.6.0 + wandb==0.23.0
                                # and clear ~/.triton/cache + /tmp/torchinductor_* (stale kernels segfault)
# torch==2.10.0 must already be installed from the CUDA-matched index-url (see README)

bash scripts/training/train_ddp.sh         # DDP path → runs stage_1_init.yaml
bash scripts/training/train_deepspeed.sh   # DeepSpeed path → runs stage_3_post.yaml (zero2)
python scripts/training/compare_yaml.py    # diff/validate configs across stages

bash scripts/inference/helios-base_t2v.sh        # released-model inference; {base,mid,distilled} × {t2v,i2v,v2v}
bash scripts/inference/experiment_interactive/helios-distilled_t2v.sh   # interactive prompt-interpolation demo
python app.py                              # Gradio demo
```

Training is **always launched through `accelerate launch train_helios.py --config <stage>.yaml`** — the `.sh` wrappers only set NCCL/torch env and assemble `ACCELERATE_ARGS` from Arnold/Metis cluster vars (`$ARNOLD_WORKER_*`, `$METIS_WORKER_0_PORT`). On a non-Arnold box you must supply `--num_processes/--num_machines/--main_process_ip/--main_process_port` yourself (or uncomment the single-GPU block in the script).

Eval is HeliosBench: `eval/` holds numbered metric scripts `0_get_aesthetic.py` … `10_merge_all_results.py`, run via `eval/run_metrics_ddp.sh`; checkpoints fetched with `eval/checkpoints/get_checkpoints.sh`.

## Architecture invariants (read before editing model/training code)

- **Chunk geometry**: `latent_window_size = 9` latent frames per autoregressive chunk = `(9-1)*4 + 1 = 33` RGB frames (Wan VAE: 4× temporal / 8× spatial downsampling). The model generates video chunk-by-chunk, conditioning each chunk on patchified history.
- **Multi-Term Memory Patchification** (`helios/modules/transformer_helios.py:1065-1068`): history latents are patchified at three granularities by separate Conv3d layers — `patch_short` k/s=(1,2,2), `patch_mid` (2,4,4), `patch_long` (4,8,8). `history_sizes = [16, 2, 1]` = long/mid/short window lengths: far history is pooled aggressively, near history stays detailed. Gated by `has_multi_term_memory_patch`.
- **Core model class**: `HeliosTransformer3DModel` (`transformer_helios.py:905`). Companion pieces in the same file: `HeliosAttention`/`HeliosAttnProcessor2_0`, `HeliosTransformerBlock`, `HeliosRotaryPosEmbed`, `LoRALinearLayer` (used by `restrict_lora`), and `Discriminator3DHead` (GAN head, line 95). Optimized kernels live in `helios/modules/helios_kernels/` (fp32 RMSNorm, triton RoPE/norm, tiled linear) and are swapped in at model-build time.
- **Scheduler**: `helios/scheduler/scheduling_helios.py:HeliosScheduler` implements the Stage-2 **Pyramid** schedule — it precomputes per-stage sigma/timestep ranges (`stages=3`, `stage_range=[0,1/3,2/3,1]`, `gamma`). Stage 1 uses plain `FlowMatchEulerDiscreteScheduler`/`UniPCMultistepScheduler` instead.
- **Everything trains as LoRA** (PEFT, `lora_layers: all-linear`, rank 128→256 across stages). The base weights stay frozen; `save_model_hook` saves only the adapter (+ norm layers if `train_norm_layers`). After training, merge with `tools/merge_lora_for_helios.py` to get the final `transformer/*.safetensors`.

## The training flow — one file, `train_helios.py` (2603 lines)

The entire pipeline is a single `main(args)` (entry at line 2454: parse `--config` → OmegaConf merge with the schema in `helios/utils/train_config.py` → big block of cross-flag assertions → `main`). All stage behavior is **config-driven branching inside one trainer**, not separate scripts. Key branch points:

**Dataloader selection** (lines 112-129) keys off `data_config`:
- `use_stage3_dataset` → `helios/dataset/dataloader_dmd.py` (multi-task: `gan_data_root` / `ode_data_root` / `text_data_root`)
- `use_stage1_dataset` → `helios/dataset/dataloader_history_latents_dist.py` (pre-encoded history latents — Stages 1 & 2)
- neither → `helios/dataset/dataloader_mp4_dist.py` (raw MP4)
Sampling is `BucketedSampler` + `StatefulDataLoader` (stateful = resumable mid-epoch).

**Loss selection** in the step loop (lines 1358-1841) — three mutually-exclusive regimes:
1. **Flow matching** (default, neither DMD nor ODE): `_flow_loss` (`utils_helios_base.py`), standard MSE flow-matching. This is Stages 1 & 2.
2. **ODE regression** (`is_use_ode_regression and is_only_ode_regression`): `_ode_regression_loss` (`utils_helios_post.py`) — regresses the few-step student onto teacher ODE trajectories. This is the *distillation init*, `stage_3_ode.yaml`.
3. **DMD2 distillation** (`is_train_dmd`): alternates generator/critic by `global_step % dfake_gen_update_ratio == 0`. `_generator_loss` + `_critic_loss` (`utils_helios_post.py`), a second `HeliosTransformer3DModel` acts as the fake-score/critic on its own `critic_accelerator`. Optional add-ons gated by their own flags: GAN (`is_use_gan`, `Discriminator3DHead` heads + R1/R2 reg), reward model (`is_use_reward_model`, `helios/videoalign/`), mean-var / smoothness / consistency regularizers, GT-history (`is_use_gt_history`). This is `stage_3_post.yaml`.

**Stage scheduling**: `is_enable_stage2` swaps in `HeliosScheduler` + `prepare_stage2_noise_input` (pyramid); else `prepare_stage1_noise_input`. `use_dynamic_shifting`/`time_shift_type` match the noise schedule to latent size and apply across all three regimes.

**EMA / checkpointing**: `use_ema` keeps a CPU copy stepped via a zero3 deepspeed config (`create_ema_zero3.py`), started at `ema_start_step`. `save_model_hook`/`load_model_hook` (lines 565-671) handle LoRA-only save/restore. `log_validation` (line 2428) runs the pipeline on `validation_prompts` and logs videos to W&B; EMA weights can be swapped in for validation.

### The 6 stage configs and how they chain (`scripts/training/configs/`)

The three *released* models (Base → Mid → Distilled) are produced by 6 configs, each chained off the previous via `transformer_model_name_or_path` + `subfolder`. Per stage there's a high-LR `_init` (fast convergence) and low-LR `_post` (refine), per the README "two-phase" note.

| config | starts from | regime | notable flags |
|---|---|---|---|
| `stage_1_init` | `Wan2.1-T2V-14B-Diffusers` | flow matching | LoRA r128, lr 5e-5, full `multi_term_memory_patch` trainable, `corrupt_history`, `is_random_drop` (v2v/t2v) |
| `stage_1_post` | `Helios-Base/transformer_init` | flow matching | LoRA r128, lr 3e-5 (refine) → **Helios-Base** |
| `stage_2_init` | `Helios-Base` | flow + pyramid | LoRA r256, lr 1e-4, `is_enable_stage2`, `is_navit_pyramid`, `stage2_sample_ratios [1,2,1]` |
| `stage_2_post` | `Helios-Mid/transformer_init` | flow + pyramid | LoRA r256, lr 3e-5, `sample_ratios [1,1,1]` → **Helios-Mid** |
| `stage_3_ode` | `Helios-Mid` | ODE regression | `is_use_ode_regression+is_only_ode_regression`, `ode_regression_weight 80`, 6 infer steps, `denoising_step_list [1000,750,500,250]`, EMA on |
| `stage_3_post` | `Helios-Distilled/transformer_ode` | DMD2 | `is_train_dmd`, `real_score_model=Helios-Base`, critic LoRA r256, `dfake_gen_update_ratio 5`, GAN scaffolding + reward model + GT history → **Helios-Distilled** |

The distilled student is **locked to 4 timesteps `{1000,750,500,250}` and is CFG-free** (`real/fake_guidance_scale`). `correct.yaml` is a patched variant from upstream issue #38 (fixes i2v train/inference mismatch, fully enables Easy Anti-Drifting) — prefer it over `stage_1_*` if reproducing the improved recipe.

### Training data is pre-encoded offline

Stages 1-3 do **not** read raw video in the hot loop — `tools/offload_data/` precomputes everything: `get-short-latents` / `get-long-latents` (VAE latents at both history granularities), `get-text-embedding` (UMT5 embeds), `get-ode-pairs` (teacher ODE trajectories for `stage_3_ode`). The dataloaders consume these `*_folders`. See `tools/offload_data/README.md` for the metadata JSON format; toy data is on HF (`BestWishYsh/HeliosBench-Weights/demo_data`).

## Inference (`infer_helios.py`, 560 lines)

One script for all tasks; the `scripts/inference/helios-{base,mid,distilled}_{t2v,i2v,v2v}.sh` wrappers just set flags. Key knobs: `--num_latent_frames_per_chunk 9`, `--is_enable_stage2` + `--pyramid_num_inference_steps_list` (Mid/pyramid), `--num_inference_steps` (50 for Base, 6 for Distilled), `--guidance_scale` (1.0 = CFG-free for Distilled). Memory/scale options: `--enable_low_vram_mode` (group offloading), `--enable_parallelism` (Ulysses/Ring/Unified context parallel), `--enable_compile`. The `experiment_interactive/` variants add `--use_interpolate_prompt` / `--interpolation_steps` / `--interpolate_time` to switch prompts mid-stream — the closest in-tree analog to the parent workspace's multi-event prompt-switching goal. A separate `helios/diffusers_version/` mirrors the model/pipeline/scheduler for the `diffusers` integration.

## Gotchas

- **Cluster**: H200 nodes on `/mnt/beegfs`, Slurm. H200s have **no NVENC** — any ffmpeg re-encode must use libx264 (CPU).
- **Stale kernel cache segfaults**: `install.sh` clears `~/.triton/cache` and `/tmp/torchinductor_*` for a reason; re-run those `rm`s after a torch/triton version change.
- **`is_train_dmd` doubles GPU memory** (generator + critic transformers, each 14B). `dmd_is_low_vram_mode`/`is_gan_low_vram_mode` swap models GPU↔CPU; the README claims up to four 14B models fit in 80 GB.
- **Config completeness**: stages share a huge flat `training_config` with many mutually-dependent flags and explicit assertions in `main` (e.g. `use_error_recycling` conflicts with `corrupt_history`/`corrupt_model_input`). Run `compare_yaml.py` and add new keys to *every* stage config, not just one.
- The `helios/videoalign/` package is a standalone VLM reward trainer (`train_reward.py`) used only when `is_use_reward_model` is set in Stage 3.

## Documentation conventions (helios-echo)

Adapted from `../Human-Replacement`'s documentation system (see its `CLAUDE.md` "实时记录规则"). Directory placement establishes document type; date prefixes establish chronology.

### Real-time recording rule (highest priority)

To survive context compaction and session restarts:
1. New discoveries / progress / disproven hypotheses → append to `logs/findings.md` / `logs/progress.md` **immediately**, not at session end.
2. Inference/eval outputs → `results/` (gitignored; generated artifacts, job logs, rendered videos).
3. Long-session implementation tracking and scratch notes → `agent-research/YYYY-MM-DD-<topic>.md` (gitignored, local only).
4. Cross-session preferences/lessons → Claude memory system, not repo files.

### Directory layout and naming

| Location | Type | Tracked | Naming |
|---|---|---|---|
| `docs/` (flat, UPPERCASE) | inherited helios-team codebase docs | yes | keep upstream `HELIOS_*.md` style; don't rename |
| `docs/specs/` | design contracts for new work (decision-first: metadata/status → scope → data contract → design → validation gates → edge cases) | yes | `YYYY-MM-DD-<kebab-topic>-design.md` |
| `docs/plans/` | executable implementation plans (numbered tasks, checkboxes, code blocks, smoke gates, final self-review) — each plan links its spec | yes | `YYYY-MM-DD-<kebab-topic>-plan.md` |
| `docs/echo-to-helios-migration-design.md` | canonical migration design (predates this convention; grandfathered) | yes | — |
| `logs/findings.md` | evidence ledger: conclusion/root-cause first, then `file:line` refs, job IDs, commits, measurements, **ruled-out hypotheses** | yes | rolling, dated sections |
| `logs/progress.md` | execution ledger: checkboxes, commits, jobs, next steps | yes | rolling, dated sections |
| `logs/research/` | curated agent research reports (deep-reads, verified design sections) | yes | `read-<topic>.md` / `design-<topic>.md` |
| `agent-research/` | the `my-docs/` role of Human-Replacement: long-session scratch, handoffs, WIP notes, and raw conversation exports (provenance only, never edit; consolidate conclusions into dated docs under `docs/`) | **no** (new files local by default; deliberately committed exports stay tracked) | scratch: `YYYY-MM-DD-<topic>.md`; exports: keep source name (`Claude_export_<title>_<uuid>.md`) |
| `results/` | generated experiment artifacts | **no** | per-run subdirs |

Deviation from Human-Replacement: they gitignore `logs/`; here `logs/*.md` are **tracked** (this repo is itself the research record) but committed at milestone granularity — append freely, commit in batches.

### Discipline rules

- **Pre-scale validation gate**: any training run or dataset job states its smoke test, sample size, acceptance criteria, and the condition blocking full-scale launch *in the plan doc* before launching. For this project the K0–K4 gates in the design doc chapter 4 are the canonical gates.
- **Kill events produce postmortems**: a killed stage/experiment gets `docs/specs/YYYY-MM-DD-<topic>-postmortem.md` (versioned status header, superseded-docs list, ruled-out table). New hypotheses must consult the ruled-out table first.
- **Review provenance**: substantial agent-generated research records model/workflow, verification methodology (e.g. adversarial verify passes), and resolved-vs-deferred issues — see `logs/findings.md` 研究方法记录 for the pattern.
- **Evidence style**: code claims cite `path/to/file.py:line`; experiment claims cite Slurm job IDs, checkpoint paths, and result dirs. Conclusion first, chronology second.
- Commit messages in English (Conventional Commits); code comments in English.

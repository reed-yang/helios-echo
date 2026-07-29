# CLAUDE.md

Guidance for Claude Code in this repository.

## This working copy: helios-echo

Working fork of `../helios-team` @ `mid_training_xiangbo`, dedicated to porting Echo-Infinity's evolving memory into Helios. Work on branch **`echo-memory`**; `mid_training_xiangbo` tracks `team/mid_training_xiangbo` (remote `team` = local `../helios-team`, `origin` = `github.com/Visko-Platform/helios-team.git`). Sync: update the sibling checkout, then `git fetch team && git rebase mid_training_xiangbo`.

Read order: ① `docs/echo-to-helios-migration-design.md`（总纲 §一 统一基线、§二 先验vs代码偏差表）② `logs/findings.md` ③ `logs/progress.md` ④ `logs/research/read-*.md`（深读报告，file:line 锚定）.

Environment: `/mnt/beegfs/yuheng/miniconda3/envs/helios` (= xiangbo's operational env; his sbatch and eval_env.sh both point at it). Tests: `PYTHONPATH=. $ENV/bin/python -m unittest discover -s tests`. GPU smokes via `srun -p gpu --gres=gpu:h200:1` on idle nodes (avoid c-node07, rwtag-pinned); Slurm H200 cluster, no NVENC (ffmpeg → libx264).

## Codebase essentials (inherited Helios; details in upstream docs/)

Helios = Wan2.1-T2V-14B re-architected into an autoregressive chunked video DiT (upstream `github.com/PKU-YuanGroup/Helios`). Driven by `accelerate launch train_helios.py --config <stage>.yaml` and `python infer_helios.py`; stage recipes and data pipeline: `docs/HELIOS_TRAINING_STAGES.md`, `tools/offload_data/README.md`.

Invariants (read before touching model/training code):
- Chunk geometry: `latent_window_size=9` latent frames/chunk = 33 RGB frames; history window `history_sizes=[16,2,1]` (+x0 anchor) = 19 latent frames, multi-term patchified by `patch_long/mid/short` Conv3d at (4,8,8)/(2,4,4)/(1,2,2) (`helios/modules/transformer_helios.py`).
- Core: `HeliosTransformer3DModel` + `HeliosAttention`/processor/`HeliosRotaryPosEmbed` in one file; optimized kernels in `helios/modules/helios_kernels/` (SDPA fallback exists; flash-attn needs CUDA + fp16/bf16).
- One trainer file, config-driven regimes: flow matching (Stage 1/2) / ODE regression / DMD2; everything trains as LoRA (PEFT all-linear) + full-rank extras via `save/load_extra_components` (`helios/utils/utils_base.py`).
- **Dual code paths**: training uses `helios/modules` + `helios/pipelines`; released inference (`infer_helios.py`) uses the separate `helios/diffusers_version` mirror — model changes need both (M1 training chain first, M2 mirror).
- Config completeness: stages share one flat `training_config` with cross-flag assertions in `train_helios.py`; add new keys to every stage config (`scripts/training/compare_yaml.py`).
- Gotchas: stale triton/inductor caches segfault (clear after torch changes); `is_train_dmd` doubles GPU memory; `HELIOS_FORCE_LR=1` resume overwrites lrs (now role-aware for memory groups).

## Evolving-memory work map (this fork's additions)

- Module: `helios/modules/helios_memory.py` (`HeliosMemoryEncoder`: 2-layer cross-attn Enc, gated EMA write, fp32 `query_state` as plain attribute — never in state_dict).
- Transformer read path: token-insertion prefix `[mem|long|mid|short|current]`, fractional RoPE ids, shared t=0 AdaLN (requires `zero_history_timestep`), per-head `memory_key_scale`, `capture_last_hidden` API (conditional 3-tuple return) — in `transformer_helios.py`.
- Trainer: config flags + `validate_evolving_memory_config` (`helios/utils/train_config.py`), extra-components section 5, role-tagged param groups, `memory_freeze_backbone` (Stage A).
- Pipeline state machine: per-section read, last-scheduled-step capture (own sigma per entry), k−2 eviction writes, `get_memory_state` — in `helios/pipelines/pipeline_helios.py`.
- Tests: `tests/test_*.py` (CPU, SDPA-patched) + `tests/smoke_real_weights.py` / `smoke_rollout.py` / `smoke_a1_assembly.py` (GPU, real weights).
- Eval toolchain (drift A/B, one driver serves all protocols): `scripts/evaluation/run_p2_interim_drift_ab.py` — `--prompt-set rep50` (structural cases, MANDATORY: raw sentences are prompt-OOD), `--segments/--sections-per-segment` (event switching via the pipeline's native interactive path), `--sections` (horizon), `--memory-partial` (load trained memory; backbone patch convs excluded by design). Launchers `sbatch_p2_r3_infer.sbatch` / `sbatch_p2_r4_eventswitch.sbatch` (both remap CVD to the max-free Default-mode GPU); metrics `tools/long_video_eval/scripts/run_helios_long_timeseries_metric.py`, batched by `sbatch_p2_metrics_batch.sbatch`. Chain evals behind training with `--dependency=afterany:<jobid>` so they survive session loss.
- Preview site: `scripts/evaluation/build_p2_site.py` → `results/p2_site/` (two pages: event-switch main + static/long-horizon sub-page). Campaigns named `p2-rep50-*` are auto-discovered on rebuild; voided raw-prompt campaigns stay excluded. Served by a detached http-server + cloudflared tunnel with keepalive (`p2_site_keepalive.sh`), current URL in `results/p2_site/.current_url`.

## Documentation conventions

Directory placement establishes document type; date prefixes establish chronology (adapted from Human-Replacement).

| Location | Type | Tracked | Naming |
|---|---|---|---|
| `docs/` (UPPERCASE flat) | inherited team docs | yes | keep upstream style |
| `docs/specs/` / `docs/plans/` | design contracts / executable plans (plan links its spec; pre-scale gates mandatory) | yes | `YYYY-MM-DD-<kebab>-design/-plan.md` |
| `logs/findings.md` / `logs/progress.md` | evidence ledger (conclusion-first, `file:line`, job IDs, ruled-out hypotheses) / execution ledger | yes | append immediately, commit in batches |
| `logs/research/` | curated agent reports & run verdicts | yes | `read-*/design-*/…-verdict-*.md` |
| `logs/session-ckpts/` | session checkpoints: single-entry context-reload snapshots (state matrix, blockers, next actions) | yes | `YYYY-MM-DD-session-checkpoint.md` |
| `agent-research/` | my-docs role: per-concept append-only ledgers (session tracking, scheduling notes); raw conversation exports (provenance, never edit) | no (new files local) | `YYYY-MM-DD-<core-concept>.md` (date = inception, for ordering); similar/ongoing topics APPEND dated sections to the existing ledger instead of new files (edits only for significant errors) |
| `results/` | generated artifacts: run logs, videos, eval outputs | no | per-run files |

Rules: kill events → dated postmortem with ruled-out table; agent research records review provenance (model/workflow/verification method); code claims cite `file:line`, experiment claims cite job IDs + paths. Commits English Conventional Commits; code comments English.

## Context stewardship rules（主 agent context 维护准则）

主 context 是决策工作区，不是信息仓库。

- **A. Context 纪律**：只承载结论与锚点、即将亲手编辑的精确片段、用户决策交互；批量信息落盘、以摘要+路径进入。红线：>500 行且非马上编辑的内容必须委托抽取；已委托获取的信息不亲自复读。
- **B. 委托纪律**：读/搜/取证/盯结果默认委托；主 agent 只保留设计仲裁、代码手术、提交、最终综合、用户交互。子代理按"结论 + file:line + 最小摘录"返回（证据卡片），全文写盘回路径。一个 worker 一个有界目标，大文件分片（递归适用）。副作用以文件系统事实验收，不信声明。
- **C. 验证纪律**：实质性变更进入昂贵验证（GPU/长任务/训练）之前，先过独立对抗审查（refute 型，拿变更去真实调用点反驳），与下一任务并行。研究期的对抗验证习惯延续到实现期。
- **D. 运行纪律**：真值通道保真——exit code/终态签名/完整日志不经过滤直达磁盘，判读放下游；>1-2 分钟任务后台化 + watcher（失败签名全覆盖，静默=失联≠成功）；有 traceback 的调试委托根因分析并行化；批次末清理陈旧任务与临时状态。

配套：模型路由见全局 `~/.claude/sub2cc/model-routing.md`（回答"派给谁"；本节回答"什么必须派、怎么派、怎么收"）。

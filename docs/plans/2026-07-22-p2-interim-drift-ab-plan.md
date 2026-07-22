# P2-Interim Drift A/B Scaffold Plan

**Spec:** design doc ch.3 §6 (评测协议: D11 external-command slope metrics, D12 ablation arms, perturbation protocol) + §7 (quantified go/no-go gates); `docs/specs/2026-07-22-real-model-test-checkpoint-decision.md` (P2 → Helios-Distilled as primary effect testbed; interim = drive distilled weights through the training-side `pipeline_helios.py` until the M2 diffusers mirror lands).

**Goal:** a turnkey drift-slope A/B loop — metric scripts first (standalone, later registered as `external_command` steps, zero core changes to the eval runner), then a paired-rollout driver on Distilled through the training-side pipeline. Validates the full plumbing and collects Distilled baseline drift curves *before* any trained-memory checkpoint exists.

**Non-goal:** effect claims. Memory is untrained (M₀ + random-init Enc), so the expected A/B outcome is ≈ no difference; the A/B here is a regression/plumbing gate and metric shakedown. Stage B/C gates from §7 apply only once trained checkpoints exist.

## Task 0 — Distilled regime mapping (gate, GPU dry-run)

- Source regime: `guidance_scale=1.0`, stage2 pyramid `pyramid_num_inference_steps_list=[2,2,2]`, `is_amplify_first_chunk` (`scripts/inference/helios-distilled_t2v.sh:13-16`). Weights: local snapshot under `/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled` (fall back to HF id `BestWishYsh/Helios-Distilled` with `HELIOS_LOCAL_FILES_ONLY=1` unset only if the local dir is incomplete).
- Map these flags to training-side `HeliosPipeline.__call__` arguments (stage2 path); produce a short param-parity table in the driver's docstring. The released script drives `infer_helios.py` (diffusers_version); we intentionally take the training-side twin per the interim decision.
- GPU dry-run: load distilled transformer dir via training-side `HeliosTransformer3DModel.from_pretrained` with memory kwargs on; 1-section latent-only generation, assert finite, assert memory module present & untouched by the checkpoint load (fresh M₀ — same expectation as the P1-B smoke).
- **Sampling-regime risk (run1 NaN lesson):** before any long rollout, run a 5-section latent-only smoke in the mapped regime, both arms finite. A regime that NaNs here is a mapping bug, not a product bug — fix the mapping, do not touch pipeline code without a root-cause report.

## Task 1 — Timeseries metric script

`tools/long_video_eval/scripts/run_helios_long_timeseries_metric.py` (new dir; standalone argparse CLI so the runner can subprocess it as an `external_command` step later — runner contract at `tools/long_video_eval/long_video_eval/runner.py:55-82, 97-106`).

- Input: one or more video paths, `--chunk_frames 33`, `--out <json>`; optional `--event_spec` JSON accepted but deferred (CLIP-over-time lands with the eventswitch phase).
- Per-chunk metrics (design ch.3 §6 "新增指标"):
  1. motion-amplitude — reuse the existing Farneback flow + scoring implementation (`eval/1_get_motion_amplitude.py:34-59, 62-69`); new part is only the 33-frame chunk-aligned series.
  2. saturation/chroma — HSV-S mean and CIELAB chroma mean per chunk; report slope + AUC.
  3. boundary LPIPS-excess — LPIPS(last frame of chunk i, first frame of chunk i+1) minus the intra-chunk adjacent-frame LPIPS baseline; series length = n_chunks − 1.
- Slope aggregation: OLS **and** Theil-Sen per metric (D11 dual-slope requirement).
- Output schema: `{video, n_chunks, per_chunk: {...}, boundary: [...], slopes: {metric: {ols, theil_sen}}, auc: {...}}`.
- CPU unit smoke: synthetic video (moving gradient, ~5 chunks) → schema complete, all slopes finite. No GPU needed (LPIPS on CPU is fine at this scale).

## Task 2 — A/B driver

`scripts/evaluation/run_p2_interim_drift_ab.py`.

- Paired rollouts, same seed + prompt per pair, via training-side `HeliosPipeline`: arm **on** = `enable_evolving_memory=True`, arm **off** = `False` (off = true no-KV: memory column never assembled).
- Decode through the pipeline VAE → mp4 via libx264 (cluster has no NVENC).
- Length: 90.75 s (2178 frames = 66 sections × 33) per vs24_long — affordable because distilled is [2,2,2]-step.
- Prompts: 2–3 drawn from the `vs24_long` prompt set (`tools/long_video_eval/configs/vs24_long.json`).
- Artifacts: `results/p2_interim/<runid>/{on,off}/*.mp4` + manifest JSON (seed, prompt, regime, wall time) + `get_memory_state()` digest per memory-on rollout (write count, queue residue, gate stats) as plumbing evidence.

## Task 3 — Validation run + verdict

- 2 prompts × 2 arms on Distilled; run the Task-1 script over all 4 videos.
- Scaffold acceptance (NOT effect gates): both arms finite end-to-end; memory-on wall-time overhead ≤ 12%; metric JSONs schema-complete; boundary series length = 65; memory writes = sections − 2 = 64.
- Verdict → `logs/research/p2-interim-baseline-verdict-2026-07-XX.md` (Distilled baseline drift curves become the reference for Stage B/C comparisons); findings entry + commit.

## Pre-scale gates (mandatory, repo convention)

1. Task 0 dry-run + 5-section regime smoke green before any 66-section rollout.
2. Task 1 synthetic-video smoke green before consuming GPU outputs.
3. Adversarial refute-review of Task 1/2 scripts (against the real runner contract and real pipeline signatures) before GPU spend — stewardship rule C.

## Execution routing

Task 1 and Task 2 implementation are bounded, code-grounded → delegate (Sol high), one task per worker, evidence-card returns. Task 0 GPU dry-run and Task 3 verdict synthesis stay with the main agent. Sequencing: starts after the rollout run2 verdict lands; independent of Stage A trainer work and may interleave with it.

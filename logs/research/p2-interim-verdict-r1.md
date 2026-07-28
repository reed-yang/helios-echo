# P2-Interim Scaffold Verdict — R1

> **【作废 VOID,2026-07-28 用户裁定】**:本判定基于 vs24_long raw 裸句 prompt,不符合结构化改写规范;结果作废,由 rep50 结构化三臂重建取代。

**Conclusion: PASS WITH CAVEAT for scaffold acceptance; NOT effect-ready.** The full Distilled plumbing/regression gate completed: all four 90.75 s videos are finite, all four Task-1 metric jobs completed with exit code 0, every JSON is schema-complete and finite with 66 chunks / 65 boundaries, and both memory-on manifests record 64 writes with queue residue `[64, 65]`. The overhead item is inherited from the clean rollout-smoke run2 measurement (+5.5%, within the 12% limit), because this run's four-way c-node08 contention invalidates its own timing comparison.

The caveat is a material A/B parity alarm, not an effect claim: memory-on mean motion is only 3.6% and 5.0% of memory-off for prompts 0 and 1 (about 28× and 20× lower). Fresh untrained M₀ was expected to produce approximately no A/B difference. The scaffold is proven executable and observable, but this divergence must be investigated before these R1 curves are used as a Stage B/C effect reference.

## Acceptance checklist

| Item | Evidence | Verdict |
|---|---|---|
| 2 prompts × 2 arms finite end-to-end | Generation jobs 5677–5680 completed `0:0`; four manifests report finite latents; four videos contain 2,178 frames at 24 fps | PASS |
| Metric JSON schema complete and finite | Metric jobs 5681–5684 completed `0:0`; required per-chunk series, four dual-slope entries, and two AUC entries parse and are recursively finite | PASS |
| Boundary series length | 65 in every JSON (66 complete 33-frame chunks) | PASS |
| Memory writes | Both on manifests: `write_count=64`, `queue_indices=[64,65]`, 66 per-section sigmas | PASS |
| Memory-on sampling overhead ≤12% | R1 timings are contention-polluted; inherited clean measurement is +5.5% in `logs/research/rollout-smoke-verdict-run2.md` | PASS (inherited); optional uncontended Distilled timing pair remains |
| Fresh-M₀ A/B sanity | Motion differs by about 20–28×; chroma/saturation remain same-order; signed boundary excess is small but changes direction | CAVEAT / investigate before effect use |

## Metric jobs and artifacts

| Job | Video | Terminal state | JSON checks |
|---|---|---|---|
| 5681 | `on/prompt_00.mp4` | `COMPLETED`, `0:0`, log `EXIT_CODE=0` | PASS: 66 chunks, 65 boundaries, finite |
| 5683 | `off/prompt_00.mp4` | `COMPLETED`, `0:0`, log `EXIT_CODE=0` | PASS: 66 chunks, 65 boundaries, finite |
| 5684 | `on/prompt_01.mp4` | `COMPLETED`, `0:0`, log `EXIT_CODE=0` | PASS: 66 chunks, 65 boundaries, finite |
| 5682 | `off/prompt_01.mp4` | `COMPLETED`, `0:0`, log `EXIT_CODE=0` | PASS: 66 chunks, 65 boundaries, finite |

Metric outputs and raw logs are under `results/p2_interim/p2-interim-r1/metrics/`. The distance backend is normalized RGB L2 for all four runs; LPIPS was unavailable and `--use_lpips` was intentionally not passed.

## Sanity comparison — descriptive only

Values are `mean / median / OLS slope / Theil–Sen slope`. These are observations of fresh untrained M₀, not evidence of memory effectiveness.

### Prompt 00 (seed 7)

| Metric | Memory on | Memory off | On/off mean |
|---|---:|---:|---:|
| Motion | 0.0376821 / 0.0181515 / -0.0012430 / -0.0007142 | 1.0576578 / 0.6955840 / +0.0359582 / +0.0311998 | 0.0356× |
| Saturation | 88.7909 / 85.4365 / -1.3188542 / -1.7416759 | 112.4032 / 102.8152 / -1.9388164 / -1.9442382 | 0.7899× |
| Chroma | 9.32560 / 7.91937 / -0.1648298 / -0.1649909 | 15.44961 / 16.16870 / -0.0816930 / -0.1053604 | 0.6036× |
| Boundary excess (L2) | 0.00057471 / 0.00015233 / +0.00003368 / -0.00000023 | -0.00044688 / -0.00015107 / +0.00000586 / -0.00000186 | signed ratio not meaningful |

### Prompt 01 (seed 8)

| Metric | Memory on | Memory off | On/off mean |
|---|---:|---:|---:|
| Motion | 0.0276423 / 0.0159450 / -0.0009108 / -0.0002656 | 0.5561381 / 0.4804102 / +0.0012170 / -0.0011172 | 0.0497× |
| Saturation | 121.9645 / 110.6756 / -1.1942158 / -1.1518934 | 104.2822 / 100.2491 / -2.3189900 / -2.4329004 | 1.1696× |
| Chroma | 25.91219 / 22.59271 / +0.3280578 / +0.1880352 | 19.23825 / 20.21125 / -0.1148230 / -0.0998967 | 1.3469× |
| Boundary excess (L2) | 0.00297660 / 0.00183006 / +0.00003098 / +0.00000569 | -0.00154448 / -0.00239354 / -0.00002186 / -0.00003158 | signed ratio not meaningful |

Saturation and chroma stay within the same order of magnitude (mean ratios 0.60–1.35). Boundary excess remains numerically small, but its signed mean and slopes differ. Motion is the wild divergence: the same qualitative suppression appears in both prompts, so it is not dismissed as a one-prompt outlier. R1 therefore establishes the metric and state-machine plumbing but is quarantined from effect interpretation pending parity diagnosis.

## Timing caveat

The manifests expose load, rollout, decode/encode, and total wall time:

| Prompt | Phase | Memory on | Memory off | Delta |
|---|---|---:|---:|---:|
| 00 | load | 194.4 s | 246.9 s | -21.2% |
| 00 | rollout | 226.9 s | 208.1 s | +9.0% |
| 00 | decode/encode | 60.7 s | 59.7 s | +1.7% |
| 00 | total | 484.3 s | 515.7 s | -6.1% |
| 01 | load | 199.5 s | 199.3 s | +0.1% |
| 01 | rollout | 773.0 s | 209.3 s | +269.3% |
| 01 | decode/encode | 113.5 s | 63.0 s | +80.2% |
| 01 | total | 1088.5 s | 472.8 s | +130.2% |

These four generation jobs shared c-node08 concurrently. Prompt 01 memory-on experienced severe contention while the other three completed, so neither total wall time nor the phase breakdown is a clean overhead measurement. No ≤12% Distilled claim is made from R1. The gate instead inherits rollout-smoke run2's uncontended memory-on 943.4 s versus off 894.6 s (+5.5%). A dedicated uncontended Distilled pair is optional follow-up; generation was not rerun for this verdict.

## Provenance

- Generation: jobs 5677 (`on/prompt_00`), 5678 (`on/prompt_01`), 5679 (`off/prompt_00`), 5680 (`off/prompt_01`), all on c-node08 and `COMPLETED 0:0`.
- Generation logs: `results/p2_full_p0_on.log`, `results/p2_full_p1_on.log`, `results/p2_full_p0_off.log`, `results/p2_full_p1_off.log`; each ends in its arm-specific `EXIT=0` marker.
- Videos/manifests: `results/p2_interim/p2-interim-r1/{on,off}/prompt_0{0,1}.{mp4,manifest.json}`.
- Metrics: jobs 5681–5684 on c-node08, each requested `--mem=32G --cpus-per-task=16 --time=01:00:00`; JSON and raw `EXIT_CODE` logs under `results/p2_interim/p2-interim-r1/metrics/`.
- Metric CLI: `tools/long_video_eval/scripts/run_helios_long_timeseries_metric.py`, default `--chunk_frames 33`, L2 backend.
- Clean overhead source: `logs/research/rollout-smoke-verdict-run2.md` (job 5552, +5.5%).

## Ruled out / not claimed

- **No effect claim:** M₀ and the encoder are fresh/untrained; R1 cannot establish anti-drift benefit or harm.
- **No LPIPS claim:** boundary excess uses the declared L2 fallback only.
- **No R1 overhead claim:** four-way node contention invalidates causal timing comparison.
- **No schema/NaN explanation for motion gap:** all four payloads parse, all series and slopes are finite, and both prompts show the same direction of motion suppression.
- **No additional generation:** this verdict consumes the verified finite Task-2 artifacts only.

## Orchestrator addendum: parity-caveat root cause (resolved, not a scaffold bug)

Timeline discriminator over the per-chunk metric series (66 chunks; writes begin at k=3):

| arm | motion chunks 1-3 | motion mid | motion tail | saturation head→tail | sat slope (theil-sen) |
|---|---|---|---|---|---|
| on p0 | 0.025/0.004/0.030 | ~0.02 | ~0.01 | 134→75 | −1.74 |
| off p0 | 0.061/0.055/0.017 | 0.34-0.58 | 0.17-0.92 | 175→57 | −1.94 |
| on p1 | 0.024/0.038/0.027 | 0.000-0.001 | ~0.03 | 177→92→142 (partial recovery) | −1.15 |
| off p1 | 0.077/0.100/0.119 | 0.48-1.20 | up to 2.69 | 175→24 | −2.43 |

1. **Pre-write chunks (1-3) are same order of magnitude across arms** → RNG pairing and the read
   path are sound; the 20-28× mean gap is NOT an early-divergence/determinism bug.
2. **Off arm = classic beyond-horizon drift**: rising frame-diff (flicker) + catastrophic
   desaturation, exactly the failure mode the design docs predict for 66-section no-KV rollouts.
3. **On arm = static collapse**: untrained memory feedback loop (open gate bias 0.75 EMA-writes
   captured hiddens; state tokens re-injected every section) pulls content into a low-motion
   attractor; desaturation is consistently SLOWER than off (and partially recovers on p1).
4. Consequence: the plan's "fresh-M0 A/B ≈ no difference" expectation only holds for early
   sections. R1 stands as the **untrained-memory long-horizon baseline** (both arms pathological
   in different modes), motivating the Stage A-C curriculum; it is not an effect reference and
   no scaffold fix is warranted.

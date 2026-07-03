# vs24 checkpoint evaluation — analysis, runtimes & recommendations

_Written 2026-06-16 after pausing the eval run. Companion to the auto-generated tables in
[`VS24_EVAL_REPORT.md`](VS24_EVAL_REPORT.md). Suite: the merged `yushen/long-video-eval`
(DOVER / PickScore / HPSv3 / VBench / HeliosBench), run on 8×H200, sharded one job per GPU._

Two checkpoint sets were evaluated:

| set | videos | length | what it is |
|---|---|---|---|
| `eval_out_vs24_long` | 7 steps × 20 | ~91 s (2178 f) | single long prompt per video |
| `eval_out_vs24_eventswitch` | 14 ckpts × 40 | ~58 s (1386 f) | prompt-**switching** (~6 prompts/video) |

`step-0` = the base model **before** this continue-train (`video_single_24fps` run); higher
steps/checkpoints = more training.

---

## 1. What completed before the pause

| phase | long | eventswitch |
|---|---|---|
| DOVER (quality) | ✅ full / start15 / end15 | ✅ full / start15 / end15 |
| PickScore (text-image) | ✅ all 3 segments | ✅ all 3 segments |
| HPSv3 (human pref) | ✅ all 3 segments | ✅ all 3 segments |
| **VBench** (7 dims) | ✅ **all 3 segments (168/168 shard-runs)** | ⏳ **partial** — only `full`, ~8/14 ckpts, subject+background dims |
| HeliosBench (aesthetic/motion/semantic + drift) | ❌ not started | ❌ not started |

So the **long set is fully evaluated** for DOVER/PickScore/HPSv3/VBench; the **eventswitch set
has the complete quality/preference triple + drift** but only partial VBench and no HeliosBench.
HeliosBench was queued behind the slow eventswitch-VBench tail and never started.

---

## 2. Results — long set (complete)

Full-video scores by training step (higher = better, except `*_raw`):

| metric | step-0 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | read |
|---|---|---|---|---|---|---|---|---|
| dover_overall | **0.578** | 0.520 | 0.486 | 0.495 | 0.440 | 0.482 | 0.468 | quality ↓ |
| dover_aesthetic | **0.696** | 0.678 | 0.625 | 0.613 | 0.576 | 0.626 | 0.595 | quality ↓ |
| imaging_quality | **64.3** | 54.1 | 54.4 | 55.5 | 52.2 | 54.4 | 53.1 | quality ↓ |
| aesthetic_quality (VBench) | **0.567** | 0.530 | 0.541 | 0.529 | 0.525 | 0.540 | 0.534 | quality ↓ |
| hpsv3_frame_mean | **9.27** | 4.32 | 5.10 | 5.38 | 4.35 | 4.89 | 4.24 | pref ↓ |
| pickscore_frame_mean | **20.78** | 19.96 | 20.31 | 20.44 | 20.33 | 20.29 | 20.12 | pref ↓ (small) |
| subject_consistency | **0.896** | 0.859 | 0.870 | 0.876 | 0.864 | 0.868 | 0.865 | ~flat ↓ |
| background_consistency | **0.928** | 0.907 | 0.921 | 0.919 | 0.914 | 0.918 | 0.919 | ~flat |
| motion_smoothness | 0.995 | 0.994 | 0.994 | 0.994 | 0.995 | 0.994 | 0.995 | flat |
| temporal_flickering | 0.990 | 0.986 | 0.987 | 0.990 | 0.990 | 0.988 | 0.990 | flat |
| **dynamic_degree** | **0.300** | **0.900** | 0.800 | 0.550 | 0.400 | 0.750 | 0.600 | **motion ↑** |

**The headline finding is a motion-vs-fidelity tradeoff.** `step-0` (base) scores highest on
*every per-frame quality/preference metric* — but its `dynamic_degree` is only **0.30**, i.e. it
generates near-**static** video. Continue-training roughly **doubles-to-triples motion**
(`dynamic_degree` 0.30 → 0.55–0.90) while per-frame fidelity drops (DOVER −0.10, imaging −10 pts,
HPSv3 −4). This is the classic interpretation: the base model "looks good per frame" partly
*because* it barely moves; training on the 24 fps dynamic corpus injected motion at some fidelity
cost. Quality metrics alone would wrongly crown the base model.

**Temporal drift** `|end15 − start15|` (lower = more stable over the 91 s clip):

| metric | most stable ckpt | note |
|---|---|---|
| dover_overall | **step-5000** (0.069) | step-4000/4500 worst (~0.11) |
| hpsv3_frame_mean | **step-5000** (0.862) | base & step-4000 worst (~1.5) |
| pickscore_frame_mean | step-0 (0.417) | trained ckpts 0.45–0.68 |

→ Among trained checkpoints, **step-5000 is the most temporally stable** on quality/preference.
Combined with its mid-pack motion (0.75) and decent quality, **step-5000 is the best all-round
trained checkpoint** on the long set; step-4000 is a runner-up on raw quality but drifts more.

## 3. Results — eventswitch set (quality complete; VBench partial)

Full-video scores across the 14 checkpoints (step-0 … checkpoint-6500):

| metric | step-0 | best trained | trained range | read |
|---|---|---|---|---|
| dover_overall | **0.498** | ckpt-4000/5500 (0.463) | 0.438–0.478 | quality ↓ then flat/noisy |
| dover_aesthetic | **0.681** | ckpt-500 (0.651) | 0.601–0.651 | ↓ then flat |
| hpsv3_frame_mean | **6.68** | ckpt-1500 (5.56) | 4.44–5.56 | pref ↓, noisy |
| pickscore_frame_mean | **19.26** | ckpt-5500 (19.19) | 18.92–19.21 | ~flat |
| subject_consistency¹ | 0.832 | — | 0.80–0.83 | slight ↓ |
| background_consistency¹ | 0.915 | — | 0.90–0.92 | slight ↓ |

¹ VBench partial: only `full` segment, first ~8 checkpoints, 2 of 7 dims — **not conclusive**.

Same direction as the long set — `step-0` highest, a drop after training begins — but **among the
trained checkpoints the curve is essentially flat/noisy** (no clean monotonic trend, no standout
"best" step). PickScore is nearly flat by design here: these are prompt-**switching** videos and the
text metrics score every frame against the **first** prompt only, so they mostly reflect the opening
event and are not a faithful measure of mid-video prompt adherence.

**Drift** is markedly higher than the long set (HPSv3 drift ~2.0–3.2 vs ~0.9–1.5), which is expected:
the prompt switches themselves cause large start→end changes, and the first-prompt scoring penalises
the end of the clip. Lowest-drift trained checkpoints: dover→ckpt-1000, hpsv3→ckpt-3500,
pickscore→ckpt-3000 — but the spread is within noise, so **no firm "best" eventswitch checkpoint** can
be claimed from the current metrics. A prompt-switch-aware metric (per-segment prompt alignment) would
be needed to rank these properly.

---

## 4. Per-metric runtime & cost (8×H200, measured)

Wall-clock from this run (sharded one process per GPU). "per (ckpt × segment)" is the unit that scales.

| metric | model / cost driver | long (20 vids) | eventswitch (40 vids) | scaling |
|---|---|---|---|---|
| **DOVER** | own clip sampler, GPU-light | ~3–5 min / segment (all ckpts) | ~14–16 min / segment | cheap; ∝ #videos |
| **PickScore** | CLIP-ViT-H, per-frame | ~1–3 min / segment | ~2–5 min / segment | **cheapest** |
| **HPSv3** | Qwen2-VL-7B reward, per-frame | ~4–13 min / segment (model-load bound) | ~7–32 min / segment | load-bound; first segment slowest |
| **VBench** (7 dims) | DINO+CLIP+RAFT+AMT+MUSIQ, **CPU frame-decode bound** | **~6.3 h total** (3 seg × 7 ckpts) → ~18 min / (ckpt×seg) | **~50 min / (ckpt×seg)** → ~12 h for `full` alone, **~35 h for all 3 segments** | **dominant**; ∝ #videos × length × dims |
| segment prep (start15/end15 re-encode) | libx264 CPU (no NVENC on H200) | ~25 min (one-time) | ~40 min (one-time) | CPU serial |
| **HeliosBench** | ViCLIP/CLIP/AMT | _not run_; expect ~10–30 min / set (similar to DOVER+PickScore) | _not run_ | — |

**Whole-set wall-clock (this run):**
- Long, DOVER+PickScore+HPSv3, 3 segments: **~30–50 min**. + VBench: **+6.3 h**.
- Eventswitch, DOVER+PickScore+HPSv3, 3 segments: **~1.5–1.75 h** (+~40 min prep). + VBench (all 3 seg): **~35 h** (the reason the run was paused).

Why VBench is ~10–70× the others: it runs **7 models per video**, decodes **every frame** of 58–91 s
clips on CPU (GPU sits near-idle, which is why utilisation looked low), and the suite runs it on
**3 segments**. PickScore/HPSv3 by contrast sample **1 frame per 33-frame chunk** (~40–66 frames total).

---

## 5. Recommendations

**For routine per-checkpoint comparison (every checkpoint, fast turnaround):**
run **DOVER + PickScore + HPSv3 + HeliosBench**. Together they cover no-reference quality, per-frame
prompt alignment, human-preference, and (HeliosBench) the purpose-built long-video aesthetic/motion +
start→end **drift** signal — all in **well under an hour for the long set, ~2 h for eventswitch**. Add
VBench's **`dynamic_degree`** specifically — it's the one dim that exposed the motion increase and is
relatively cheap on its own.

**Treat VBench as an occasional, scoped run, not per-checkpoint:**
- Run it on the **`full` segment only** (drop start15/end15) → cuts cost ~3×. VBench's start/end drift
  added little here and is the least cost-effective slice.
- Run it on a **subset of checkpoints** (e.g. base + best-by-cheap-metrics + final), not all 14.
- Full 7-dim VBench on the eventswitch set as configured (~35 h) is **not worth it per-checkpoint**;
  reserve it for a final 2–3 candidate models.

**Metric suitability reminders (see report methodology):**
- Compare **full** for level and **start15-vs-end15 drift** for stability; don't compare full vs
  segment absolute values for PickScore/HPSv3 (different frame sampling).
- For **eventswitch**, PickScore/HPSv3/semantic only see the **first prompt** → they don't measure
  mid-video prompt-switch adherence. To rank prompt-switching quality, build a **per-segment prompt
  alignment** metric (slice each video at event boundaries, score each segment vs its own prompt).
- VideoAlign stays excluded (short-video reward model; ill-defined at 60–90 s).

**Model-selection takeaway from what's done:**
- Quality metrics alone favour `step-0`, but that's an artifact of its near-static output
  (`dynamic_degree` 0.30). Judge with motion in the loop.
- On the long set, **step-5000** is the best-rounded trained checkpoint (lowest quality/pref drift,
  mid motion, competitive quality). step-4000 trades a bit more drift for slightly higher raw quality.
- On the eventswitch set, the trained checkpoints are within noise of each other on the current
  (prompt-unaware) metrics — no firm winner; needs the per-segment metric above.

---

## 6. How to resume / finish

Everything is staged and idempotent. From `helios-team/`:

```bash
source /mnt/beegfs/xiangbo/helios_runs/eval_env.sh   # env overlay + paths

# HeliosBench (fast, still missing for both sets):
bash /mnt/beegfs/xiangbo/helios_runs/run_heliosbench.sh long \
  "step-0 step-3000 step-3500 step-4000 step-4500 step-5000 step-5500" 2178
bash /mnt/beegfs/xiangbo/helios_runs/run_heliosbench.sh eventswitch \
  "step-0 checkpoint-500 ... checkpoint-6500" 1386

# Finish eventswitch VBench — recommend full-segment only (edit SEGMENTS=full) or fewer ckpts.
# DOVER/PickScore/HPSv3 are already done and will be skipped (SKIP_DONE_METRICS=1).

# Refresh the tables report after any run:
python scripts/evaluation/make_eval_report.py
```

Outputs live in `helios_runs/eval_metrics/vs24_{long,eventswitch}/`; normalized inputs in
`helios_runs/eval_norm/`; drivers + master in `helios_runs/`. Full setup notes are in the memory
file `vs24-long-video-eval-suite.md` (suite was incomplete upstream — `common.py`/`io_utils.py` were
reconstructed and an HPSv3 transformers-5 shim added; these fixes are in `scripts/evaluation/`).

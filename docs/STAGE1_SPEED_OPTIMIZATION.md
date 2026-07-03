# Stage-1 LoRA training: hyperparameter alignment + speed optimization (368×640, 388k corpus)

This doc covers two things requested together:
1. **Aligning** the 368×640 continue-train to the official **Stage-1-init** hyperparameters (what was wrong, what changed).
2. **Optimizing training speed without losing accuracy** — finding what actually limits throughput and fixing it.

All numbers are measured on this cluster (H200×8/node, InfiniBand NDR), `helios` env, the 387,670-clip
`latents_cfr_int_30b_368x640` corpus (a 6,000-clip symlink subset for the micro-benchmarks). Config:
`scripts/training/configs/stage1_init_cfr368.yaml`. Bench harness: `scripts/training/bench_speed_{1node,4node}.sbatch`.

---

## 0. TL;DR (中文)

- **之前训练的问题（相对 Stage-1-init）**：用的是 continue-train（post 风格）的超参 —— `lr 3e-5`（应为 `5e-5`）、
  `constant_with_warmup`+warmup 500（应为纯 `constant`、无 warmup）、global batch 64/96（应为 128）、steps 17000（init 为 5500）。
  history-corrupt 已经和仓库里 `stage_1_init.yaml` 一致；按你给的 spec（p_b=0.8 noise / p_c=0.1 downsample）我在新 config 里启用了
  downsample（仓库参考是 noise-only，已注释可一键还原）。
- **真正限制速度的不是数据/IO**：dataloader 每个 batch 只要 ~6ms，beegfs 随机读 65ms/文件、292 文件/秒/节点。
  GPU 利用率只有 ~30–40%，每步 ~4s 计算 + ~8–13s 同步空转。
- **干净的、无损精度的提速**：
  - `free_memory()`（每步 `gc.collect()+empty_cache()`）→ 限频：**17.57 → 13.06 s/it（−26%）**。这是唯一干净、确定的提速。
- **gradient_checkpointing 必须开**：关掉在 bs2 就 OOM（~138 GiB）。
- **`find_unused_parameters=False` 不可取**：random_drop 下要配零值 touch 才不死锁，而 touch（对所有可训练参数求和+all-reduce）
  本身的开销 ≥ 它省下的——单机持平、4 节点更慢（45 → 56 s/step）。保持默认 `find_unused=True`。
- **多机 ~20s/step 的 all-reduce 开销是真实的，但不是互联问题**：NCCL 已在用 IB + GPUDirect RDMA（8×400G）。
  它是 LoRA 数百个小梯度张量的延迟受限 all-reduce；进一步优化（梯度分桶/减少 LoRA 目标层）超出"无损精度快速优化"的范围。

---

## 1. Hyperparameter alignment vs the official Stage-1-init spec

The spec you provided (Stage-1-init): GB 128, AdamW(0.9, 0.999, 1e-8, wd 1e-4), **lr 5e-5**, **constant**, **5.5k steps**,
grad-clip 1.0, **LoRA r128/α128**, bf16; history corrupt p_a=0/p_b=0.8/p_c=0.1/p_d=0.1, b∈[0,0.33], c∈[0,0.1].

| item | your previous 368 run | Stage-1-init spec | aligned config |
|---|---|---|---|
| learning rate | **3e-5** (post-style) | **5e-5** | 5e-5 |
| lr schedule | constant_with_warmup, warmup 500 | **constant** (no warmup) | `constant`, warmup 0 |
| global batch | **64 / 96** (4/6-node, bs2) | **128** | 4 nodes × 8 × **bs4** × accum1 = 128 |
| max_train_steps | 17000 (continue) | **5500** | 5500 |
| LoRA r/α | 128 / 128 ✓ | 128 / 128 | 128 / 128 |
| optimizer/clip/precision | AdamW, 1.0, bf16 ✓ | same | same |
| corrupt_model_input (p_a) | false (=0) ✓ | 0.0 | false |
| history corrupt | mode="noise": clean 0.1 / noise(σ≤0.33) 0.9 | p_b=0.8 noise, **p_c=0.1 downsample**, p_d=0.1 clean | mode="random", prob 0.8889 → 0.8/0.1/0.1 |

**What was "wrong":** the previous run was a deliberate *continue-train* (post LR 3e-5 from checkpoint-9000), so vs a faithful
*init* it differed on lr, schedule, global batch and step count. The corruption block already matched the authors'
`stage_1_init.yaml` exactly (noise-only). The only spec-vs-repo discrepancy is the downsample branch: the spec says p_c=0.1
downsample, the repo reference uses noise-only. The aligned config follows **your spec** (downsample on) with a one-line comment
to revert to the repo reference.

Mapping the spec's corrupt probabilities to the code (`corrupt_history_latents`, utils_helios_base.py:275):
`P(clean)=noise_corrupt_clean_prob=0.1` (p_d); with `corrupt_mode="random"`, `P(noise)=0.9·noise_mode_prob` and
`P(downsample)=0.9·(1−noise_mode_prob)` → set `noise_mode_prob=0.8/0.9=0.8889` to get p_b=0.8 / p_c=0.1. Noise σ∼U[0,0.33]
matches b; downsample resize fraction U[0.9,1.0] matches "c strength ≤0.1".

---

## 2. What limits training speed (it is NOT data/IO)

Measured the real steady-state first (correcting an earlier mis-read of a tqdm artifact):

| config | steady | GPU util |
|---|---|---|
| 1 node, bs2, gc-on (stock) | **~17.6 s/it** | ~30–40% |
| 4 nodes, bs2/bs4, gc-on (stock) | **~28 s/it (gb64) / ~45 s/it (gb128)** | ~30–48% |

GPU util ~30–40% ⇒ each step is **~4 s of compute + ~8–13 s of synchronized idle** (all GPUs go 100%→1% together).
So the limiter is a **per-step stall**, not the GPUs being busy. Ruled out by direct measurement:

| candidate | measurement | verdict |
|---|---|---|
| dataloader | bs2 batch delivered every **6 ms** (70 samples/s/rank) — `dataonly_probe.py` | **not it** |
| raw beegfs IO | **65 ms/file**, 292 files/s/node (3.4 GB/s); a step needs ~16 files | **not it** |
| optimized kernels | FP32-RMSNorm / Flash-RoPE / Flash-LayerNorm/RMSNorm already patched in at build | already on |
| interconnect | NCCL on **IB + GPUDirect RDMA**, 8×400 Gb/s HCAs | already optimal |

The stall is in the **training-loop / DDP**, not the data path.

---

## 3. Speed levers measured (1 node, 8×H200, gc-on bs2 unless noted)

| lever | s/it | Δ | accuracy | keep? |
|---|---|---|---|---|
| **A** stock (free_memory on, find_unused on) | 17.57 | — | — | baseline |
| **B** `free_memory()` throttled (skip per-step gc.collect+empty_cache) | **13.06** | **−26%** | identical | **YES** |
| gradient_checkpointing OFF | **OOM** (~138 GiB at bs2) | — | identical | NO (must keep gc on) |
| find_unused=OFF + zero-touch (1 node) | ~12 | ~−10% gross, **wash** (touch costs ≈ saving) | identical | NO (slower at 4-node, §4) |
| bs4 (gc-on), per-sample | 32.99 s/it @ bs4 (peak ~90 GiB) | ≈ bs2 per-sample | identical | use for gb128 |

**`free_memory()`** is called every step twice (utils_helios_base.py:174 and train_helios.py:1915); each call runs
`gc.collect()` + `torch.cuda.empty_cache()`, which defeats the CUDA caching allocator and stalls the loop. Throttling is
mathematically identical training. Gated behind `HELIOS_THROTTLE_FREE=1` (default off → no behavior change unless opted in).

**gradient_checkpointing** must stay ON — disabling it OOMs (the 14B model with seq≈10,600 tokens stores too much activation
at bs2). This is why we hit gb128 via **bs4** (peak ~90 GiB, fits) rather than bs8.

---

## 4. Multi-node (2→4 node): the all-reduce / `find_unused` finding

For the **same per-GPU batch (bs4)**, going 1→4 nodes adds **~20 s/step** — and it is **not** bandwidth (NCCL already uses
**IB + GPUDirect RDMA across 8×400 Gb/s HCAs**; verified in the NCCL_DEBUG=INFO log). It is the **latency-bound all-reduce of
hundreds of small LoRA gradient tensors** across nodes plus per-step barriers.

I tested the obvious lever — **`find_unused_parameters=False`** (it defers/serializes reduction less). It is **required** to be
`True` today because `is_random_drop` t2v/v2v microbatches leave some history params ungraded (DDP otherwise **deadlocks** —
confirmed it hangs). I added a **zero-touch** (`loss += 0.0 * sum(p.sum() for trainable p)`, extended from the full-finetune path)
to make `find_unused=False` safe. Result: it did **not** help — the touch (summing + all-reducing *every* trainable param each step)
costs more than find_unused saves.

| config (4 nodes, gb128 bs4, free-throttled) | s/step | verdict |
|---|---|---|
| find_unused=ON (default) | **~45** | keep |
| find_unused=OFF + zero-touch | **~56** | rejected (touch overhead > saving) |

So **keep `find_unused=True`**. Reducing the multi-node all-reduce further (gradient bucketing tuning, fewer LoRA target modules,
or ZeRO) is possible but goes beyond accuracy-neutral quick wins. NCCL/IB needs no tuning (already optimal).

---

## 5. Recommended optimal config (accuracy-neutral)

- Config: `scripts/training/configs/stage1_init_cfr368.yaml` (gb128 via bs4×accum1, gc-on, lr 5e-5 constant, 5500 steps).
- Launcher env (set in `scripts/training/train_stage1_init_cfr368.sbatch`):
  - `HELIOS_THROTTLE_FREE=1` — skip the per-step `free_memory()` (**−26%**, identical training). The proven win.
- Keep `find_unused_parameters=True` (default — find_unused=False+touch tested slower at multi-node).
- Keep `gradient_checkpointing: true` (mandatory), keep NCCL defaults (IB+GDR already optimal).

The `HELIOS_THROTTLE_FREE` knob is **mathematically identical** to stock training (only memory-pool management changes) — it
changes *speed*, not the learning signal. The `HELIOS_DDP_FIND_UNUSED` / zero-touch gates exist in code (utils_helios_base.py,
train_helios.py:147) but default OFF — they were the experiment, not the recommendation.

## 6. Do NOT change
- `gradient_checkpointing` → off (OOM).
- `find_unused=False` **without** the zero-touch (DDP deadlock under is_random_drop).
- `max_train_steps` on an existing `output_dir` without first deleting `output_dir/config.json` (config-mismatch guard).

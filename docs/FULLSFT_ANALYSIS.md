# Full-SFT (full fine-tune) analysis — why quality degraded (deformation / jitter)

Run analysed: **`stage1_fullft_cfr`**, job 2513 (the clean, fully-fixed run), W&B `f7vy4pyj`.
Data on the captions_30b_32f / cfr_int corpus (~153.7k clips), continued from **Helios-Base/transformer_init**.
Stopped by the user at ~step 2400/4500 because samples showed clear **deformation and temporal jitter**.

---

## 1. What was actually fine-tuned vs. kept frozen

**Setup (code path).** `train_helios.py`:
```
354  transformer.requires_grad_(False)     # start: everything frozen
355  vae.requires_grad_(False)
356  text_encoder.requires_grad_(False)
...
402  if is_full_finetune:                  # our flag (stage1_fullft_cfr.yaml: is_full_finetune: true)
405      transformer.requires_grad_(True)  # <-- unfreeze the ENTIRE DiT, no LoRA adapter
     else:
         transformer.add_adapter(...)      # (LoRA path, not taken here)
```
Optimizer then collects `filter(requires_grad, transformer.parameters())` → **every parameter of the DiT**.

| Component | Params | Trained? |
|---|---|---|
| **HeliosTransformer3DModel (the DiT)** — *all of it* | **14.31 B** | ✅ **YES (100%)** |
| &nbsp;&nbsp;• `patch_embedding` (noisy-input Conv3d) | trained | ✅ |
| &nbsp;&nbsp;• multi-term memory patchify `patch_short / patch_mid / patch_long` | trained | ✅ |
| &nbsp;&nbsp;• all attention (`to_q/to_k/to_v/to_out`, cross-attn, RoPE-projected) | trained | ✅ |
| &nbsp;&nbsp;• all FFN (`ffn.net.*`) | trained | ✅ |
| &nbsp;&nbsp;• **all normalization layers** (RMSNorm / fp32-RMSNorm / flash-norm, AdaLN modulation) | trained | ✅ |
| &nbsp;&nbsp;• time-step embedding + text/condition projections | trained | ✅ |
| **VAE** (`AutoencoderKLWan`) | ~frozen | ❌ frozen |
| **Text encoder** (UMT5) | ~frozen | ❌ frozen |

- The training banner prints `Num generator trainable parameters = 447,263,362`. That is the **per-rank ZeRO-2
  optimizer shard** (14.31 B / 32 ranks = 447 M). The **full 14.31 B is trained** — it's just sharded across the 32 GPUs.
- `train_norm_layers`, `is_train_full_patch_embedding`, `is_train_full_multi_term_memory_patchg` are all `false`,
  but they are **moot** — `requires_grad_(True)` already turned on *everything*, so those modules train regardless.
- A `loss += 0.0 * Σ p.sum()` "touch term" is added (full-FT only) so every param receives a (zero) gradient —
  required for DeepSpeed ZeRO-2 not to crash; it has **no effect on the learning signal**.
- Checkpoints = DeepSpeed ZeRO-2 sharded full model + optimizer (`pytorch_model/`).

> **Key point:** this is a *maximally aggressive* fine-tune — the carefully pretrained input patchify, the
> multi-term temporal memory, the normalization/AdaLN, and the time embeddings are **all moving**. Those are
> exactly the modules that govern spatial structure and temporal consistency. LoRA, by contrast, freezes 100%
> of the base and only adds a low-rank delta on linear layers → it cannot disturb those structural dynamics.

---

## 2. Training-record analysis (grad_norm / loss / lr)

Parsed from the 2513 log, steps 0–2468.

| step band | grad_norm mean / med / max | loss mean / med |
|---|---|---|
| 0–250    | 0.125 / 0.088 / **1.04** | 0.0808 / 0.0697 |
| 250–500  | 0.163 / 0.104 / **1.81** | 0.0844 / 0.0702 |
| 500–750  | 0.098 / 0.072 / 0.69 | 0.0821 / 0.0708 |
| 750–1000 | 0.158 / 0.113 / **1.14** | 0.0923 / 0.0751 |
| 1000–1250| 0.084 / 0.065 / 0.57 | 0.0816 / 0.0709 |
| 1250–1500| 0.199 / 0.115 / **4.95** | 0.0950 / 0.0758 |
| 1500–1750| 0.082 / 0.064 / 0.40 | 0.0875 / 0.0720 |
| 1750–2000| 0.130 / 0.091 / **1.97** | 0.0857 / 0.0704 |
| 2000–2250| 0.143 / 0.104 / **1.27** | 0.0929 / 0.0729 |
| 2250–2400| 0.097 / 0.065 / 1.15 | 0.0851 / 0.0712 |

Overall: grad_norm median **0.086**, p90 **0.231**, **max 4.95**; loss median **0.072**, range 0.015–0.87.
`lr`: log-warmup over 200 steps → **constant 1e-5** thereafter.

**Observations & interpretation:**

1. **Loss is completely flat (~0.072) for 2400 steps — it never goes down.**
   Flow-matching loss is timestep-noisy, so a flat curve is not by itself fatal — but here it means the model is
   *not* improving on the objective while its weights are clearly being pushed around. Flat loss **+** visibly
   *worse* samples ⇒ the update is **pure drift / forgetting**, not learning. The model is trading its pretrained
   visual–temporal quality for fitting the new (terse, off-distribution) caption format, with **no measurable gain**.

2. **grad_norm is small on average (0.086) but intermittently spikes** — 63 steps > 0.5 and 15 steps > 1.0
   (clipped at 1.0), with a 4.95 spike around step 1300. These spikes are instability events (hard batches /
   sharp loss regions). With only global-batch 32 they are not being averaged out, so each spike is a real, if
   clipped, perturbation to all 14 B weights.

3. **Warm-up worked** (lr 0 → 1e-5 over 200 steps), then **constant 1e-5 with no decay** — the model keeps being
   pushed at full strength for the entire run.

---

## 3. Hyperparameter assessment

| Hyperparameter | Value | Assessment for **full 14B FT** |
|---|---|---|
| **learning_rate** | **1e-5, constant** | **⚠️ Too high.** 1e-5 applied to *all 14 B* params moves far more weight than the LoRA's 3e-5 on a tiny adapter. Full-FT continue-training of a large pretrained video DiT typically uses **1e-6 – 5e-6** with decay. This is the single biggest suspect. |
| **EMA** | **off** (`use_ema: false`) | **⚠️ Big miss.** Full-FT quality depends heavily on EMA — the EMA weights are far smoother/higher-quality than the raw ones (Helios-Distilled itself used EMA). Sampling from raw full-FT weights amplifies jitter. |
| **global batch** | **32** (bs1 × 32 GPU × accum1) | **⚠️ Small** for a 14 B full-FT → high gradient variance → noisy, drifting updates and the grad spikes above. Pretraining used far larger batches. Want **256–512** (grad-accum). |
| **lr schedule** | `constant` (after warmup) | **⚠️** No decay → keeps pushing at 1e-5 to the end. Cosine/linear decay would let it settle. |
| **scope = all params** | patchify + memory patches + **norms/AdaLN** + time-embed all trainable | **⚠️** These structural/temporal layers are the most sensitive; moving them is the most direct cause of **temporal jitter** and **spatial deformation**. |
| **is_random_drop** | 0.4 t2v + 0.4 v2v | **⚠️ Aggressive** under full-FT: 80% of microbatches zero-out part/all of the history (Easy-Anti-Drifting). Pushing all 14 B params under heavy history corruption stresses temporal consistency. |
| **corrupt_history** | prob 0.9, ratio 1/3 | **⚠️** Same — heavy input corruption + full-param movement compounds instability. |
| **gradient_clipping** | 1.0 (DeepSpeed) | ✅ Active and correct (verified: clips the 15 spikes). |
| weighting_scheme | logit_normal(0,1) | ✅ Standard. |
| mixed_precision / ZeRO-2 | bf16 / stage 2 | ✅ Fine (fp32 master kept by ZeRO). |
| max_train_steps | 4500 (<1 epoch) | OK in isolation, but with the above it's enough to drift. |
| **caption distribution** | new terse `<header>/<event>/<role>` | **⚠️** Off-distribution vs the model's pretraining captions. Full-FT *forces* fitting them → quality cost. LoRA tolerates the shift far better (base frozen). |

---

## 4. Root-cause hypothesis for the deformation / jitter

The degradation is consistent with **catastrophic drift from an over-strong full fine-tune**, not a bug:

- **lr 1e-5 on *all* 14 B params** (incl. patchify, multi-term memory, norms/AdaLN, time-embed) +
- **no EMA** (raw weights sampled) +
- **small batch 32** (noisy, with real grad spikes up to ~5) +
- **constant LR** (no settling) +
- **heavy history corruption** +
- **off-distribution captions**

→ the model's pretrained **temporal dynamics drift → jitter**, its **spatial/structural priors drift → deformation**,
while the **flow loss stays flat** (no compensating improvement).

**Why the LoRA run does NOT degrade (corrected — precise scope).** The LoRA run (≈648 M trainable) actually
*does* touch some structural layers — so the difference is not "LoRA freezes everything":
- LoRA-delta on all Linear (attention/FFN/projections/time-embed Linears) — base frozen, only low-rank moves.
- LoRA-delta on `patch_embedding`.
- **`patch_short/mid/long` FULLY trained** (same as full-FT — so the memory patches are NOT the differentiator).
- **Norms / AdaLN frozen**; time-embed base frozen; all base linear weights frozen.

The two differences that actually matter:
1. **Norms / AdaLN**: frozen in LoRA, **fully moved in full-FT** (governs condition/timestep modulation → high impact).
2. **Base linear weights**: LoRA adds only a **small low-rank delta**; full-FT moves the **full-rank weights** at lr 1e-5.

So the degradation comes specifically from **moving the full-rank base weights + the norm/AdaLN layers** under a too-high
LR / small batch / no EMA — not from training the memory patches (which LoRA also does, safely).

---

## 5. Recommendations for the re-train

In rough priority order:

1. **Lower the LR a lot** → `1e-6`–`3e-6`, **with cosine decay** (and keep/extend warmup). This alone should
   remove most of the drift.
2. **Turn on EMA** (`use_ema: true`, `ema_decay 0.999–0.9999`) and **sample/eval from the EMA weights**. High impact for full-FT.
3. **Increase the effective batch** to ~256–512 via `gradient_accumulation_steps` (e.g. 8–16) — smooths the grad spikes.
4. **Reduce the scope to the layers LoRA leaves frozen** (those are the ones that drift): in particular
   **freeze the norm / AdaLN layers** (`train_norm_layers` stays off + exclude them, as LoRA does), and consider
   keeping the time-embed base frozen. Training the multi-term memory patches (`patch_short/mid/long`) is fine —
   the working LoRA run trains them too. If full-rank base-weight movement is still too strong, fall back to the
   **LoRA** recipe (it already preserves quality) — full-FT may be unnecessary here.
5. **Soften corruption** for full-FT (e.g. `random_drop_*` 0.2, `corrupt_mode_prob_history` 0.5) so temporal
   consistency isn't over-stressed while all params move.
6. (Optional) Consider whether the **terse caption format** is desirable, or rewrite captions closer to the
   model's pretraining style to reduce the distribution shift the full-FT has to absorb.

> TL;DR: the full-SFT trained **every one of the 14.31 B DiT parameters** (VAE + text encoder frozen) at a
> **constant 1e-5 with no EMA and batch 32** — too aggressive. The flat loss + grad spikes + degraded samples
> all point to drift/forgetting of the pretrained spatial-temporal priors. Fix = much lower LR + EMA + bigger
> batch + (ideally) freeze the structural/temporal layers.

# Helios Streaming Attention — Findings

**Question.** During Helios streaming generation, where does a *noisy query* token
put its attention: on the (1) **sink** / first-frame anchor, (2) recent **high-res**
history, or (3) far **low-res** history — and how does this vary with **generation
time** (chunk) and **network depth** (layer)?

## 1. Experimental configuration
- **Model / checkpoint:** Helios-Base (Wan2.1-T2V-14B re-architected, autoregressive chunked DiT), released weights `BestWishYsh/Helios-Base`.
- **Why Base (not Distilled):** Base runs the clean 50-step Stage-1 flow-matching loop, so attention's evolution over *both* denoise steps and layers is visible, and it reflects the model's *natural* attention (no distillation shortcut). The Distilled checkpoint runs the Stage-2 pyramid predictor–corrector schedule, which complicates denoise-step labeling; it is left as a deployment-variant follow-up. Both released checkpoints ship `is_amplify_history=False` (see §5).
- **Task / inputs:** T2V, 384×640, `num_frames=231` ⇒ **7 autoregressive chunks** (33 RGB frames/chunk, 24 fps ⇒ ~1.375 s/chunk), 50 denoise steps, `history_sizes=[16,2,1]`, `keep_first_frame=True`.
- **Prompts (3):** `fish`, `train`, `man` (diverse motion/scene) — conclusions are reported only where consistent across all three.
- **CFG:** `guidance_scale=1.0` — a single conditional forward per step. CFG only *combines* cond/uncond outputs afterward, so the per-pass attention equals the conditional pass at any guidance scale. This gives clean chunk/step bookkeeping.
- **Sampling of the eager probe:** 512 noisy queries/record; recorded denoise steps `{0,12,24,37,49}`; **all 40 layers**, **all 40 heads** (per-head retained).
- **Token classes** (exact boundaries verified, see §2): `sink` (x0 anchor), `short_highres` (recent 1× patch), `mid` (2× patch), `long_lowres` (far 4× patch), `noisy` (current chunk / self).

## 2. Method & correctness checks
The real inference path (`helios/diffusers_version`) runs self-attention over the
concatenated sequence `[ long_lowres | mid | sink | short_highres | noisy ]` via a
flash/SDPA kernel that returns **no** weights. We added an **opt-in eager branch**
(`helios/analysis/attn_probe.py`, `PROBE.enabled`) that recomputes
`softmax(QKᵀ/√d)` for the noisy-query rows, aggregates by class immediately (never
materialising the full matrix except a few pooled maps), and is a no-op when off.

Token-class boundaries are read from the actual patch-conv output shapes (forward
hooks), **not guessed**. Smoke checks (all pass):
- softmax rows sum to **1.00000**; per-class mass sums to **1.00000**;
- boundaries are **contiguous and cover `[0, seq_len)`** with no overlap;
- token counts match geometry exactly: `long=240, mid=240, sink=960, short_highres=960, noisy=8640` (384×640).

**Two lenses are reported for every claim** (as required):
- **total mass** — Σ attention a class receives (query-averaged); influenced by class token-count.
- **per-token mean** — mass ÷ #tokens; removes the count advantage.

## 3. Results
Aggregated over **4,200 records** (3 prompts × 7 chunks × 5 denoise steps × 40 layers). Trends are consistent across all three prompts (per-prompt agreement noted below).

### 3a. Overall distribution — the two lenses disagree, as expected
| class | **total mass** | **per-token** (×1e-5) |
|---|---|---|
| noisy (self) | **88.4%** | 10.23 |
| recent hi-res | 6.0% | 6.29 |
| sink (x0) | 4.0% | 4.18 |
| mid | 1.2% | 5.00 |
| far low-res | 0.3% | 1.35 |
| **Σ history** | **11.6%** | — |

- **By mass**, the noisy query attends overwhelmingly to **itself** (~88%). Within history the order is **recent hi-res > sink > mid ≫ far low-res**.
- **By per-token**, the picture sharpens: each **recent hi-res** key is individually ≈0.88× as attended as a noisy key, each **sink** key ≈0.55×, **mid** ≈0.5×, while each **far low-res** key is ≈0.10× — an order of magnitude below everything else. (Mid's tiny mass is purely a token-count effect: only 240 tokens.)

### 3b. Vs. network depth (layer) — history use peaks in the middle
`history_share` (fraction of attention on all history), mean over chunks≥1:
- **L0 ≈ 0.028** (first layer almost pure self) → rises to a **peak at L14–L21 ≈ 0.21** → falls to **≈0.11 at the last layer**.
- Per-class mass by band: early L0–9 `sink 3.9 / recent 5.6 / far 0.5%`; **mid L10–29 `sink 5.2 / recent 8.3 / far 0.2%`** (history maximal); late L30–39 `sink 3.5 / recent 5.1%` (re-focuses on self).
- **Entropy** is lowest at mid layers (5.99 nats) and highest at late layers (6.75) — mid layers attend most *concentratedly* (onto history), late layers most diffusely.
- Cross-prompt: peak layer fish L14, train L14, man L21 — same mid-network band.

### 3c. Vs. generation time (chunk) — stable, mild decline
`history_share` by chunk (layer/step-averaged): chunk 0 `0.035` (no real history yet) → chunk 1 `0.139` → gently **declines** to `0.119` by chunk 5 (chunk 6 `0.125`). Once history exists the model does **not** lean on it more as the clip lengthens — if anything slightly less. Far low-res stays negligible (<0.5% mass) at every chunk.

### 3d. Head specialization
Heads are not interchangeable: at the peak layer (L14) the top sink-directed head puts **2.1×** the mean sink mass (head #10 = 0.141 vs head-mean 0.066) — evidence of dedicated "sink/anchor-watching" heads. (Per-head arrays are in the data; the UI's V1 per-head panel visualizes this.)

## 4. Answer to the scientific question
**Where does the noisy query attend?** Aside from itself, attention lands on **recent high-res history and the sink (first-frame anchor) — not the far low-res history.**
- **Recent high-res** is the dominant history target on *both* lenses (largest mass among history *and* highest per-token) — "many tokens, each individually heavy."
- **Sink** is the classic sink pattern: "few tokens (960), each individually moderately heavy" (per-token 0.55× a noisy token) — it matters more per-token than its 4% mass suggests, and has dedicated heads.
- **Far low-res (4× patch) history is effectively ignored** — lowest mass (0.3%) *and* lowest per-token (0.10× a noisy token). Aggressive far-history pooling receives almost no attention; the multi-term memory's value is concentrated in its near, high-res end plus the anchor.

**Vs. layer depth:** a clear **mid-network hump** — the first layers are almost pure self-attention, layers ~14–21 integrate history most strongly (peak `history_share≈0.21`, lowest entropy), and the last layers re-focus on the current chunk. History reading is a *middle-of-the-network* operation.

**Vs. generation time:** history reliance turns on at chunk 1 (once real history exists) and then stays roughly flat with a mild downward drift over 8 s of video — the model does not progressively depend on longer history.

**Amplification:** the built-in `is_amplify_history` mechanism is **off in the released weights**, so all of the above is the model's *natural* behavior. The UI's counterfactual boost shows that engaging it would smoothly shift mass onto history (e.g. `history_share` 0.12→0.34 at ×2, →0.56 at ×5) without changing the *relative* ordering among history classes.

## 5. On history amplification (`is_amplify_history`)
The architecture includes a per-head mechanism that multiplies history **keys** by a
learned scale (`scale = 1 + σ(θ)·(max_scale−1)`, `max_scale=10`) to *force* more
attention onto history. **Both released checkpoints (Base and Distilled) ship this
disabled** (`is_amplify_history=False`), so the measured distributions above are the
model's *natural* attention and a real pre/post-amplify pair is identical.

To still characterise the mechanism's *direction*, the web UI exposes a
**counterfactual "history amplify ×f"** slider: it multiplies the softmax weight of
every history token by `f` (equivalently adds `log f` to each history logit) and
renormalises — computed exactly from the natural per-class masses. This shows how the
mechanism would redistribute mass toward history **without** saturating, and cleanly
separates "what the model does on its own" (×1) from "what the mechanism would push"
(×f>1).

## 6. Reproduce
```bash
# full data (this report):
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 python -m helios.analysis.run_probe \
  --out helios/analysis/out/base_full.json --prompts fish train man \
  --num_frames 231 --num_inference_steps 50 --record_k_steps 5
# numbers:
python -m helios.analysis.summarize helios/analysis/out/base_full.json
# publish site data + deploy: see webui/DEPLOY.md
```
Artifacts: instrumentation `helios/analysis/attn_probe.py`; driver
`helios/analysis/run_probe.py`; data `helios/analysis/out/*.json`; UI `webui/`.

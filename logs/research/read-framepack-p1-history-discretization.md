# FramePack-P1 History Discretization — What It Is, Precisely

**Provenance**: researched by Terra (GPT-5.6 Terra high-effort worker), 2026-07-29, via WebSearch + WebFetch against primary sources (arXiv abstract/HTML pages, official project page); no local repo access, no code execution.

## 结论

1. History Discretization = **train-time replacement of history-frame latents by their nearest K-means codebook centroid** (Eq. 6: `Q(F)_p = argmin_k ||F_p − Ω_k||_2`, codebook `Ω∈ℝ^{K×C}` fit by K-means over a whole precomputed-latents dataset). It is applied to the **conditioning/history latent frames** (after VAE encode, before being consumed as context), **not** to the noisy target being denoised.
2. It is **not training-free**: "We replace all history frames F with Ω_Q(F) **during training**" — the diffusion backbone must be finetuned to consume quantized history. Only the codebook fitting (K-means) is offline/non-learned; the model itself needs retraining.
3. Default/best K reported inconsistently across the same paper: text says **K=128** ("strong drift reduction with relatively minimal training difficulties"), but the architecture-comparison Table 2 row is labeled **K=256**. No K sweep table is shown (deferred to unreleased supplementary).
4. Reported effect: consistently reduces the 4 drift deltas (Clarity/Motion/Semantic/Anatomy) vs. matched vanilla baseline across ~10 packing-schedule configs, and raises human ELO from ~1030–1092 to ~1139–1225. It is competitive with but slightly behind "inverted anti-drifting sampling" on the drift metrics, while retaining a **larger dynamic-range** (motion) score.
5. This is a **separate mechanism from Helios's existing FramePack-style patchify compression** (`patch_long/mid/short`, which is the original "packing" scheme, i.e. context-length compression by kernel size). History discretization is an additive P1-era technique layered on top of packing; it has never had a public code/checkpoint release (open GitHub issue, unanswered, asking for it).
6. **Verdict**: a training-free port of history discretization to Helios is **not plausible as literally described** — it requires finetuning to be effective and no public code/weights exist to transplant. A cheap "as-if" proxy (post-hoc round history latents to a fixed grid or externally-fit codebook, without finetuning) is untested by the authors and is a genuinely different intervention, not a reimplementation of the paper's method.

---

## 1. Source identification and provenance

- Primary paper: arXiv **2504.12626**, "Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models" (Zhang, Cai, Li, Wetzstein, Agrawala). Abstract page: https://arxiv.org/abs/2504.12626 — submitted 17 Apr 2025 (v1), revised 21 Apr 2025 (v2), **revised again 14 Oct 2025 (v3, current)**. The v3 HTML (https://arxiv.org/html/2504.12626v3) is the version that contains the History Discretization material (Eq. 6, Table 1/2 ablations). The paper's displayed title matches exactly; the string "FramePack-P1" itself does **not** appear in the arXiv page text — the P1 branding is used only on the project page, which cites this same arXiv ID plus a NeurIPS 2025 entry under the identical title, confirming v3 of 2504.12626 is the "P1" content (not a separate paper).
- Official project/results page: https://lllyasviel.github.io/frame_pack_gitpage/p1/ — announced designs, gallery of qualitative results, **no code, no equations, no hyperparameters**. It explicitly cites the arXiv paper above as where the "implementation specifics" live.
- Code status: **no P1 code/weights released**. GitHub issue `lllyasviel/FramePack#738`, "FramePack-p1 Did you come out?" (opened 16 Aug 2025), remains open/unassigned as of the search date. The main repo (`github.com/lllyasviel/FramePack`) only ships the original FramePack and FramePack-F1 (released 3 May 2025); P1 is preview-only.
- Secondary/aggregator sources consulted (used only for triangulation, not as evidence of technical content): Scrapbox `work4ai/FramePack-P1`, "Deep Paper" summary of 2504.12626. Marked UNCONFIRMED where used; none contributed facts beyond what the primary sources state.

## 2. Where in the pipeline it applies (Required Evidence #1)

- Operates on **latent** frames: the paper's dataset definition is "a dataset Φ∈ℝ(B×T×H×W)×C of precomputed latent videos"; quantization is per-latent-pixel across channel dim C. The paper states generally "All definitions of frames and pixels refer to latent representations, as most modern models operate in latent space." So: **after VAE encode**, before any patchify/kernel-compression step that FramePack's packing scheme applies.
- It replaces the **history/conditioning context frames**, not the noisy section being denoised: "We replace all history frames 𝑭 with Ω_Q(𝑭) **during training**."
- The paper does **not** explicitly state the inference-time insertion point beyond this training-side substitution (i.e., it doesn't spell out whether at inference the model quantizes its own rolled-forward history the same way, though this is the obvious intended reading given the training substitution and Table 1's "vanilla sampling with discrete history" as an inference-time sampling variant). It also does **not** discuss any KV-cache interaction.
- No statement about whether it applies per-chunk/per-section uniformly or only to specific history tiers (long/mid/short granularity levels in the packing schedule) — Table 2's row is `f16k4f2k2f1k1_g9+D, K=256`, i.e. discretization is applied on top of one particular packing schedule, but the paper doesn't say discretization is tier-selective vs. applied uniformly across all three granularities.

## 3. Exact operation — math / pseudocode (Required Evidence #2)

Codebook construction (offline, non-learned, one-shot over the training set):
- Run **K-means** over all latent pixels of the precomputed latent-video dataset Φ ∈ ℝ^{(B×T×H×W)×C} → codebook **Ω ∈ ℝ^{K×C}**, K a free positive-integer hyperparameter. Quoted: history discretization "converts all history to discretization tokens (directly apply K-Mean to the entire dataset)."

Quantization (paper's Eq. 6), nearest-centroid assignment in L2:

```
Q(F)_p = argmin_k || F_p − Ω_k ||_2
```

- `p` indexes a latent-pixel position (spatio-temporal location in the T×H×W grid).
- `Q(F) ∈ {0,...,K−1}^{T×H×W}` — an index map into the codebook.
- Reinjection / "dequantize": `Ω_{Q(F)} ∈ ℝ^{T×H×W×C}` — each latent pixel is replaced by its assigned centroid vector, and this is what is fed into the network as the history/context tensor in place of the original continuous `F`.

Net pipeline: `VAE-encode → per-pixel nearest-centroid lookup against a pre-fit global codebook → replace latent value with that centroid → feed as history context (train-time forced; presumably inference-time too)`.

Rationale quoted: "discrete autoregressive systems (e.g., LLMs) often demonstrate less obvious drifting than continuous autoregressive systems" and "drifting can be mitigated if the history representation is not sensitive enough to distinguish between the training and inference frames" — i.e., quantization collapses the (train-time perfect vs. inference-time model-generated) latent distributions onto the same finite discrete support, removing the fine-grained distributional gap that the model could otherwise learn to be sensitive to.

## 4. Hyperparameters and defaults (Required Evidence #3)

- Only stated hyperparameter: codebook size **K** (cluster count). No bit-width, no temperature, no separate per-tier K values are given.
- Text default: **"In our tests, K=128 gives strong drift reduction with relatively minimal training difficulties."**
- Table 2 (cross-method comparison) instead reports the discretization row at **K=256** ("f16k4f2k2f1k1_g9+D, K=256") — the paper is internally inconsistent about which K is "the" recommended default; treat K∈{128,256} as the reported operating range, not a single confirmed default.
- Trade-off stated: "higher K giving stronger anti-drifting effects but also more challenging learning for smooth transitions between sections."
- Degenerate limits given explicitly: at **K=1**, "the history becomes meaningless as one single color"; as **K→∞**, "the effect is equivalent to no discretization" (recovers vanilla continuous history).
- A full K sweep is deferred: "We provide more detailed ablations for K in the supplementary materials" — that supplementary was not locatable (not attached to the arXiv HTML, not linked from the project page).

## 5. Training-free or requires training? (Required Evidence #4)

**Requires training — settled explicitly, not inferred.** Quotes:
- "We replace all history frames 𝑭 with Ω_Q(𝑭) **during training**."
- "We also present an anti-drifting **training method** to convert frame history into discrete tokens so as to reduce the history disparity between training and inference."
- Naming convention confirms it's a trained variant, not a sampling-time trick: "The notation **+d** means converting history into discrete space" (contrast with `_x_f1k1` suffix = anti-drifting sampling variants, which are inference-time-only and applied to an already-trained model).

What is still training-free / reusable without retraining: only the **codebook-fitting procedure** (K-means over a latent dataset) is itself non-learned/offline and could in principle be run against any latent dataset without touching the diffusion weights — but using the resulting codebook to quantize history fed into an **already-trained, non-quantization-aware** model is not what the paper does or validates; the backbone must be finetuned on quantized history to realize the reported gains. No sentence in the source claims or tests drop-in application to an existing checkpoint.

## 6. Ablation / quantitative effect on drift (Required Evidence #5)

Drift metric definition (Eq. 7): `Δ_drift^M(V) = |M(V_start) − M(V_end)|`, `V_start` = first 15% of frames, `V_end` = last 15% of frames, absolute value used to be direction-agnostic (generation order can be forward or backward). Four instantiations of M: **Clarity** (MUSIQ/SPAQ, artifacts/blur), **Motion** (VBench-modified frame-interpolation smoothness), **Semantic** (ViCLIP video-text score), **Anatomy** (VBench ViT detector on hands/faces/bodies). Lower Δ = better (less drift). Human eval: pairwise A/B, 100 results/ablation, **ELO-K32**; paper notes "ELO differences within ±16 are considered ties."

Matched vanilla vs. vanilla+D pairs (same packing schedule; Δ in %, lower=better; from Table 1, representative rows):

| Config | ΔClarity | ΔMotion | ΔSemantic | ΔAnatomy | ELO |
|---|---|---|---|---|---|
| td_f16k4f2k2f1k1_g9 | 3.18 | 3.42 | 7.45 | 18.05 | 1074 |
| td_f16k4f2k2f1k1_g9**+D** | 2.30 | 2.49 | 4.12 | 14.11 | 1222 |
| td_f8k8f4k4f2k2f1k1_g9 | 3.25 | 3.45 | 7.45 | 16.56 | 1090 |
| td_f8k8f4k4f2k2f1k1_g9**+D** | 2.43 | 2.74 | 4.45 | 14.37 | 1225 |

(Full 10-pair table extracted during research; every +D row improves or ties every metric vs. its vanilla counterpart; ELO rises from the 1030–1092 band to 1139–1225.)

Cross-strategy positioning (Table 1/2): **Inverted anti-drifting sampling** wins the drift metrics outright (e.g. ΔClarity≈2.18–2.35, ΔMotion≈1.77–2.15, ELO 1220–1232) but "has a relatively small dynamic range" (i.e., suppresses motion). **Vanilla+discrete history (K=256)** is reported as "very competitive in drifting measurements, while having a relatively larger dynamic range" (Table 2: ΔClarity 3.13, ΔMotion 2.05, ΔSemantic 2.89, ΔAnatomy 8.74, Dynamic 91.74%, ELO 1224 — a statistical tie with inverted anti-drifting's ELO 1220 under the paper's own ±16 tie rule). So the paper's headline claim is: discretization ≈ matches inverted anti-drifting on human preference while preserving more motion dynamism, at the cost of a training-time modification.

## 7. Interaction with other FramePack anti-drift ingredients / version differences (Required Evidence #6)

- **Original FramePack** (context "packing": geometrically-compressed patchify kernels for older history) targets the **forgetting** problem (fixed-length context regardless of video length) — this is exactly Helios's existing `patch_long/mid/short` mechanism and is unrelated to discretization.
- **Anti-drifting sampling / inverted anti-drifting sampling**: inference-time-only sampling-order tricks (generate an endpoint far away first, or generate in reverse order from a fixed endpoint) — orthogonal, composable with discretization; comparisons in Table 1 treat them as separate axes (`_x_f1k1` suffix = anti-drifting sampling, `f1k1_x_..._t{a,c,d}` = inverted).
- **Planned Anti-Drifting** (the other P1-era design, paired with History Discretization on the P1 project page): predicts far-away sections before near ones, addressing drift *between* planned endpoints; History Discretization addresses drift *at/over* the endpoints themselves ("the endpoints themselves will not drift"). The two are explicitly framed as complementary, not alternatives.
- **History noise augmentation** (Diffusion Forcing-style per-frame noise levels on history): not discussed as a direct comparison point in this paper's ablation tables; it is a distinct line of work (Diffusion Forcing, CausVid) that the broader literature treats as a *different* mechanism for the same exposure-bias problem (see §2 of this report's second task below) — no interaction/ablation between the two is reported here.
- **FramePack-F1**: not mentioned in this paper at all (F1 is a separate, code-released variant focused on forward-only/simplified sampling, not discretization); the source material gives no basis for a direct technical comparison of F1 vs. P1's discretization.
- P1 branding itself only exists on the project page; the arXiv v3 paper is the technical backing but does not use "P1" terminology, meaning there is exactly one authoritative technical description (this paper) and one non-technical results/marketing page.

## 8. Reimplementation feasibility assessment

- **Fully specified and reimplementable as literally described**: the codebook-fit (K-means over latent dataset) and the quantization operator (Eq. 6, nearest-centroid replace) are precise enough to code directly — this part needs no further source reading.
- **Not fully specified**: (a) which K to use as a single default (128 vs. 256 conflict, no sweep table available), (b) whether discretization applies to all three packing tiers (long/mid/short) or a subset, (c) whether/how it's re-applied at inference to the model's own rolled-forward history vs. only forced during training, (d) K-means practical details (initialization, iteration count, sample-count/subsampling of the latent dataset, whether codebook is shared across channels/tiers or per-tier).
- **Blocking issue for "training-free" reuse on Helios**: the paper's own method requires finetuning the backbone to consume quantized history; there is no evidence it works if bolted onto an already-trained checkpoint without retraining. No public code or weights exist to transplant (open, unanswered GitHub issue as of search date). Therefore a literal, training-free port is not supported by the public record — at best, an engineer could (1) implement Eq. 6's quantization purely as a novel *inference-time-only* intervention to test empirically whether Helios's *already-trained* memory/history channels tolerate discretization without retraining (untested by the original authors, would need its own ablation), or (2) treat this as a training-time recipe to adopt in a future finetune, mirroring the paper's actual setup.

---

## Second task: neighbouring training-free anti-drift techniques for AR/AR-diffusion video

| Technique | Mechanism (one line) | Training-free? | Reported to help drift specifically? | Citation |
|---|---|---|---|---|
| Deep Sink + Participative Compression ("Deep Forcing") | Reserve half the sliding attention window for persistent, RoPE-realigned "sink" tokens (Deep Sink) + importance-aware KV-cache pruning that discards degraded/stale history (Participative Compression); explicitly built to need **no finetuning** | Yes (authors' explicit design goal) | Yes — targets drift, temporal repetition, and motion deceleration directly | arXiv:2512.05081, "Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression" |
| Frequency-aware RoPE Modulation + Antiphase Noise Sampling + inference-only Attention Sink ("FLEX") | Rebalances 3D-RoPE frequency exposure at inference and perturbs noise-initialization phase to counter two identified extrapolation failure modes, all without touching weights | Yes | Yes — framed as a horizon-extension/drift fix beyond training length | arXiv:2602.14027, "Train Short, Inference Long: Training-free Horizon Extension for Autoregressive Video Generation" |
| Manifold-constrained Tweedie matching ("FlowLong") | At each inference step, projects the denoiser's estimate back onto the (learned) data manifold via a Tweedie-formula-based constraint, rather than trusting raw rolled-forward context | Yes (inference-time only) | Yes — proposed specifically against drift/motion repetition in KV-cache-reliant AR video models (FramePack, Self-Forcing++, Rolling Forcing named as the class it targets) | arXiv:2605.20910, "FlowLong: Inference-time Long Video Generation via Manifold-constrained Tweedie Matching" |
| VAE decode→re-encode of history (round-trip projection) | Periodically decode drifted latents to pixels and re-encode, snapping accumulated latent error back toward the VAE's valid manifold before reusing as context | Yes (applied as a post-hoc/appendix comparison, no retraining) | Mixed/partial — appendix explicitly finds the *baseline* (CausVid-style extrapolation) "suffers severe color drift and visual artifacts during segment stitching"; the re-encode variant is offered as the fix, but this is reported only as an appendix-level ablation, not the paper's main contribution — treat quantitative strength as UNCONFIRMED pending direct read of that appendix | arXiv:2508.03334, "Macro-from-Micro Planning for High-Quality and Parallelized Autoregressive Long Video Generation" (appendix) |
| History noise augmentation (add Gaussian noise to history/context frames before conditioning) | Perturbs history frames with noise to make the model robust to context distribution mismatch; **origin is Diffusion Forcing's training scheme**, but light-weight noise injection has also been tried as a pure inference-time knob on already-trained diffusion-forcing-style models | Partially — the technique's *training* form (Diffusion Forcing) is not training-free; an inference-only noise-injection variant is training-free but reported to be the **weakest** option in a direct ablation | Reported to help somewhat but a direct A/B found it ranked lowest vs. alternatives (resampling), attributed to noise not matching the model's actual inference-time error mode | arXiv:2504.12626 (FramePack, ablates history-frame noise as a baseline design point) and BAgger, arXiv:2512.12080 ("Backwards Aggregation for Mitigating Drift..."), which explicitly frames noise-augmented context as leaving "context perturbations" mismatched from real inference-time drift; also cross-checked against a direct ablation in "End-to-End Training for Autoregressive Video Diffusion via Self-Resampling" (arXiv:2512.15702) where noise augmentation ranked below autoregressive resampling |
| Anchor-frame / History Guidance-style CFG rescheduling over history | Apply classifier-free-guidance-like differential weighting (different masks/noise/guidance scales) between "keep consistent with history" and "diverge for motion" signals, or reweight guidance strength as a function of chunk index / distance from an anchor | Partially — the *guidance mechanism itself* (mask-and-reweight at sampling time) is training-free once the base model was trained with the corresponding flexible/masked conditioning, so it is not fully drop-in on an arbitrary checkpoint, but no additional finetuning is needed at deployment time | Mixed — BAgger explicitly notes History Guidance "can worsen error accumulation, since the model is guided to stay consistent even with already-drifted frames," i.e. it can help *forgetting* while hurting *drift* in some regimes | "History-Guided Video Diffusion" arXiv:2502.06764, and BAgger arXiv:2512.12080 for the caveat about drift interaction |

Note on scope: several of the most load-bearing anti-drift techniques in this space (Diffusion Forcing noise scheduling, CausVid/Self-Forcing causal distillation, History Discretization itself) are fundamentally training-time interventions; the table above deliberately foregrounds the subset that authors explicitly label training-free, and flags the partial cases rather than overstating training-freeness.

# Stage-1 forward pass, attention structure, and the full training pipeline (formulas + pseudocode)

This doc continues [`STAGE1_DATALOADER_FRAMES.md`](./STAGE1_DATALOADER_FRAMES.md) (chunk geometry, 19+9, x0,
multi-term compression) and explains **how attention is wired once tokens enter the transformer**, plus the
**complete pipeline from offline encoding to the flow-matching loss**. One example threads the whole thing:
a **161-frame clip → 4 chunks, 368×640, `choice_idx=3`, batch=1**.

All `file:line` references point into this repo (`helios/modules/transformer_helios.py`,
`helios/utils/utils_helios_base.py`, `helios/dataset/dataloader_history_latents_dist.py`).

---

## 1. Constants & notation

```
latent_window_size W = 9          # latent frames in one chunk
history_sizes = [16, 2, 1]        # long / mid / 1x,  sum = 19
VAE: 4× temporal / 8× spatial (causal)   # RGB = (L-1)*4 + 1
368×640 → latent 46×80, C=16
σ ∈ (0,1) flow-matching noise level;  ε ~ N(0, I);  D = transformer hidden dim
```

---

## 2. Attention structure: `restrict_self_attn` (asymmetric "history is read-only")

The core is `HeliosAttnProcessor2_0` (`transformer_helios.py:201-444`). The token sequence entering each block
splits into **two groups** (NOT three peer chunks):

| group | content | tokens (368×640) |
|---|---|---|
| **history** | long16 + mid2 + short(x0+1), the three tiers patchified and **concatenated into one** history sequence | 240+240+1840 = **2320** |
| **target** | the chunk being denoised (9 latent frames) | **8280** |

After patchify the three tiers are **concatenated into a single history sequence** (the code keeps only one
`history_seq_len`); they are told apart by their RoPE positions. Each block splits q/k/v into a `history` part and a
`target` part (`:302-304`) and runs **two independent attentions**:

```
(a) history self-attention (:365-379)
    h_out = Attn(q_history, k_history, v_history)
    → history tokens attend ONLY among themselves; they never see the target
    → independent of current noise/timestep ⇒ computed once on step 0, cached (:383-388), reused every denoise step

(b) target attention (:372-410)
    k' = cat(k_history, k_target)        # :372
    v' = cat(v_history, v_target)        # :373
    t_out = Attn(q_target, k', v')       # :410  —— NO causal mask
    → target is internally full-bidirectional (9-frame 3D) AND reads all of history
```

Asymmetric wiring:

```
            ┌──────── history (long|mid|short) ────────┐   ┌──── target chunk ────┐
history Q:  └─→ history only (bidirectional)           ┘   (cannot see target)
target  Q:  ──→ reads history ─────────────────────────────→ + internally bidirectional ←──────
```

**So "do the 3 chunks do bidirectional self-attention together?" — precisely:**

- **Within the target's 9 frames (8280 tokens): full bidirectional 3D self-attention** (space+time, **no** per-frame
  causal mask; `attn_varlen_func` is plain full attention, no `is_causal`).
- **target → history: reads it** (one-way, history as extra KV).
- **history → target: cannot read it** (history is read-only clean conditioning).
- **among the three history tiers: also bidirectional**, but self-contained.

So it is **not "three peers, all fully bidirectional"** — it is "**history is a self-contained bidirectional block,
target is bidirectional and additionally reads history, history is blind to the target**".

### 2.1 Why this design (it is not arbitrary)

"history cannot see the target" is the **linchpin of the whole real-time autoregression**:

1. **History K/V are independent of noise/timestep** → across 4/20/50 denoise steps, **history KV is computed once on
   step 0 and reused** (`self.kv_cache`, `:383-388`; `use_cache` branch `:329-331`). This is where 19.5 FPS comes from.
2. **Autoregression is at chunk level, not token level**: generating chunk-by-chunk, the previous chunk becomes the
   next chunk's history KV; *within* a chunk it is a single full-bidirectional diffusion denoise (better quality than
   token-level causal).
3. `is_amplify_history` (`:394-408`) can scale history keys by `scale_key` to **amplify/attenuate history influence**
   (an anti-drift knob).
4. `restrict_lora` (`:306-309`) attaches separate LoRA to history q/k/v, so the "history path" and "generation path"
   learn different projections.

Each block also has a separate **cross-attention to text** (`enable_cross`, the prompt's UMT5 embedding) — distinct
from the spatio-temporal self-attention above.

---

## 3. The full pipeline (formulas + pseudocode), walking the example

### 3.0 Offline encoding (one-time, `get_short-latents.py`)

```
clip(161 RGB frames, 368×640)
  N = floor(161/33) = 4
  vae_latent = VAE.encode → (N=4, C=16, 9, 46, 80)      # 4 chunks
  saved:  {uttid}_161_368_640.pt
```

### 3.1 dataloader draws one sample (`prepare_stage1_latent`, dataloader:136-196)

```
flat = rearrange(vae_latent, "n c t h w -> c (n t) h w")       # timeline of 36 latent frames
continue = cat([ zeros(C,19,46,80), flat ], dim=1)             # prepend 19 zeros → length 55
choice_idx = 3                       # this example: chunk3
s = choice_idx * 9 = 27
history = continue[:, 27:46]         # (16,2,1) all real: chunk0[8] ++ chunk1 ++ chunk2 → (C,19,46,80)
target  = continue[:, 46:55]         # = chunk3                                          → (C, 9,46,80)
x0      = flat[:, 0:1]               # clip first frame (real; choice_idx≠0 so not zeroed) → (C, 1,46,80)
```
> If `choice_idx==0`: `history` all zeros and `x0←0` (cold start).

### 3.2 split 3 tiers + x0 anchor (`prepare_stage1_clean_input_from_latents`:619-659)

```
long, mid, h1x = history.split([16,2,1], dim=1)
short_stream   = cat([x0, h1x], dim=1)        # x0 + newest 1 frame = 2 frames  ← "short is 2 frames"
```

### 3.3 add noise — TARGET ONLY (`prepare_stage1_noise_input`:734-856)

```
u  ~ density(weighting_scheme);  idx = floor(u·1000)
σ  = sigmas[idx];   timestep = σ·1000                  # scalar per sample  :763,:787
ε  ~ randn_like(target)
# (optional) corrupt_history: light noise/downsample augmentation per tier (:816)  ← history, NOT target
noisy_target = (1-σ)·target + σ·ε                      # (C,9,46,80)   :855
v_target     = ε - target                              # flow-matching velocity label  :856
```
**Key: history (long/mid/short) and x0 stay clean**; only the target is noised.

### 3.4 patchify → tokens (three Conv3d + main patch_embedding, transformer:1066-1068)

mid/long spatial 46 is `pad_for_3d_conv`-padded to 48, then divided by the stride:

```
short = patch_short(short_stream) stride(1,2,2) → ( 2,23,40) = 1840 tokens  (D dim)
mid   = patch_mid (mid)           stride(2,4,4) → ( 1,12,20) =  240 tokens   (46→48)
long  = patch_long(long)          stride(4,8,8) → ( 4, 6,10) =  240 tokens   (46→48)
hist  = cat([long, mid, short])   → history_seq_len = 2320 tokens
tgt   = patch_embedding(noisy_target) stride(1,2,2) → (9,23,40) = 8280 tokens (= query)
seq   = cat([hist, tgt])          → 10600 tokens   (+ per-token RoPE)
```

### 3.5 transformer, L blocks (`restrict_self_attn`, see §2)

```
for block in blocks:                    # L layers
    q,k,v = proj(seq);  RoPE(q,k)
    q_h,q_t = split(q,[2320,8280]); same for k_h/k_t, v_h/v_t

    h_out = Attn(q_h, k_h, v_h)                       # (a) history self-attn, cached on step 0
    k' = cat([k_h,k_t]); v' = cat([v_h,v_t])          # (b) target reads history + self
    (optional amplify: k_h *= scale_key)              # :394-408
    t_out = Attn(q_t, k', v')                         # no causal mask

    seq = cat([h_out, t_out])
    seq = seq + CrossAttn(seq, text_emb)              # prompt (UMT5)
    seq = seq + FFN(seq)
```

### 3.6 output + loss (`proj_out` → unpatchify → `_flow_loss`:20-91)

```
v_pred = unpatchify(proj_out(t_out))                  # only the 8280 target tokens → (C,9,46,80)
loss   = mean_over_target( w(σ) · (v_pred - v_target)² )      # :86-88,  v_target = ε - target
# history / x0 do NOT enter the loss (clean conditioning, not a prediction target)
```

---

## 4. Shape-flow overview (this example)

```
(4,16,9,46,80)         vae_latent
   │ flatten+pad19 + choice_idx=3
   ├── history (16,2,1)=19 frames ──split──> long16  mid2  1x1
   │                                          │     │    └┐
   │                                          │     │   x0(1) ┘→ short_stream(2)
   ├── x0(1 frame) ──────────────────────────┘     │
   └── target chunk3 (9 frames) ──(1-σ)x+σε──> noisy_target,  v_target=ε-x
                                               │
   patchify:   long→240   mid→240   short→1840  │  target→8280
                └──── hist 2320 ────┘            │
                          cat ──────────────> seq 10600
                          │  × L blocks (history self-attn[cached] ‖ target reads-history+self ‖ cross-attn text ‖ FFN)
                          ▼
                  v_pred(8280) → unpatchify (16,9,46,80)
                  loss = w(σ)·‖v_pred − (ε−x_target)‖²
```

## 5. One-line summary

One chunk is the target, the 19 frames before it are history, the clip's first frame is the x0 anchor; only the
target gets flow-matching noise `(1-σ)x+σε` with velocity label `ε-x`; the three history tiers are compressed to
240/240/1840 tokens as **clean, cacheable, read-only** conditioning; in every block **history is a self-contained
bidirectional block, the target is internally bidirectional and additionally reads history, and history is blind to
the target**, followed by text cross-attention and an FFN; the loss regresses velocity on the target only.
At inference this becomes chunk-by-chunk: the previous chunk's output → the next chunk's history KV, σ denoises over
`{…}` steps, and history KV is cached/reused → streaming real-time generation.

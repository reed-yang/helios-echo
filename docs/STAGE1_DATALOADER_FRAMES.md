# Stage-1 history-latents dataloader: frames, chunks, and how a clip is consumed

This doc explains exactly how `helios/dataset/dataloader_history_latents_dist.py`
(`BucketedFeatureDataset`) turns a pre-encoded clip into training samples, why clips must be
**≥121 frames**, and what the model actually sees each step. Worked examples for a **3-chunk (121-frame)**
and a **7-chunk (261-frame)** clip are at the end.

All `file:line` references are to this repo.

---

## 1. The unit of everything: a *chunk*

- `latent_window_size = 9` latent frames = `(9-1)*4 + 1 = 33` RGB frames (Wan VAE: 4× temporal downsample).
- The offline encoder (`tools/offload_data/get_short-latents.py:207-215`) cuts a clip into
  **`N = floor(num_frames / 33)` chunks**, encoding each 33-RGB-frame chunk into 9 latent frames.
- A `.pt` file therefore stores `vae_latent` of shape **`(N, 16, 9, H/8, W/8)`** — `N` chunks × 16 VAE
  channels × 9 latent frames × spatial. (Empirically verified: 121→N=3, 161→4, 221→6, 261→7, …)

So a clip is **not** trained as one long sequence — it is a *stack of `N` independent 9-frame chunks*.

## 2. What one training sample is (fixed size, regardless of clip length)

Each `__getitem__` returns exactly **one (history, target) pair** built by `prepare_stage1_latent`
(dataloader_history_latents_dist.py:136-196):

- **target = 1 chunk = 9 latent frames** (the block to denoise this step).
- **history = `sum(history_sizes) = 16+2+1 = 19` latent frames** (the memory context just before the target).
- **section = history + target = 19 + 9 = 28 latent frames.**

28 contiguous latent frames decoded back to RGB = `(28-1)*4 + 1 = ` **109 RGB frames** — this is where the
number "109" comes from. **It is a fixed per-step training window, NOT the clip length.** A longer clip just
gives *more chunks to sample a target from*; each individual sample is always this 28-latent (≈109-RGB) window.

The 19 history frames are further split by recency into the **Multi-Term Memory** granularities
`history_sizes = [16, 2, 1]` = **long / mid / short** and patchified by separate Conv3d layers
(`transformer_helios.py:1065-1068`, `patch_long` stride 8, `patch_mid` 4, `patch_short` 2): the 16 *oldest*
history frames are pooled most aggressively (coarse far memory), the 1 *newest* stays detailed. The dataloader
hands over the contiguous 19-frame block; the split/patchify happens in the model.

## 3. How `prepare_stage1_latent` builds the pair (the mechanism)

```
source = vae_latent (N, 16, 9, h, w)
flatten chunks ->            timeline of  N*9  latent frames        # rearrange "b c t h w -> c (b t) h w"
prepend 19 ZERO latent frames -> continue = [19 zeros] ++ [N*9 real]   # length 19 + N*9
choice_idx ~ Uniform{0 .. N-1}                                       # pick which chunk is the target
start = choice_idx * 9
history = continue[start        : start + 19]                        # the 19 frames before the target
target  = continue[start + 19   : start + 28]                        # = real chunk #choice_idx (9 frames)
```

Key consequences of the **19-frame zero-pad**:
- `choice_idx = 0` → history is **all zeros** (simulates the *start of generation*, no past).
- early `choice_idx` → history is **partly zero-padded** (the model only has a little past).
- once `choice_idx ≥ 3` → history is **fully real** (3 chunks ≈ 27 ≥ 19 frames of real past exist before it).

`is_keep_x0=True`: the very first latent frame (x0) is also returned; if `choice_idx==0` it is zeroed
(no real anchor frame at generation start). This is the "Easy Anti-Drifting" first-frame handling.

## 4. Why clips must be ≥121 frames (= ≥3 chunks)

`floor(121/33) = 3`. The filter requires **≥3 chunks**, enforced in two places:
- **Before encoding:** `scripts/data_prep/build_offload_json_cfr.py:91` (`--min-frames 121`) drops short clips.
- **At train time (redundant safety):** `dataloader_history_latents_dist.py:103` `if num_frame < 121: continue`
  (`num_frame` parsed from the `.pt` filename `{uttid}_{num_frame}_{H}_{W}.pt`).

Why ≥3 chunks specifically:
- `num_rollout_sections` defaults to **3** (dataloader_history_latents_dist.py:20). The
  **rollout / DMD-teacher-forcing / GAN** path (`return_all_vae_latent=True`) takes a *contiguous*
  `19 + num_rollout_sections*9 = 46`-frame window and **hard-asserts `total_sections ≥ num_rollout_sections`**,
  raising `ValueError("Not enough sections")` otherwise (dataloader:178-182). So the *encoded data must
  guarantee ≥3 chunks* for those Stage-3 regimes to run on it.
  - In a plain flow-matching LoRA run, `return_all_vae_latent` is `False` (it's gated on
    `dmd_teacher_forcing or is_use_gan`, train_helios.py:803-810), so that hard path isn't taken — but the
    121 filter is still applied (it's baked into the shared dataloader), and ≥3 chunks also guarantees the
    random `choice_idx` can land on zero / partial / fuller-history cases.
- **121 vs the bare 99 floor:** the encoder buckets each clip to a length bucket (`…141, 121, 101, 81…`,
  `dataloader_mp4_dist.py`). Both 101 and 121 yield 3 chunks, but 121 (=3.67 chunks) is a conservative margin
  over the borderline 101 (=3.06) — robust to a few frames lost in decode/cut. The filter keeps buckets ≥121.

## 5. Worked example A — a **121-frame clip (N = 3 chunks)**

`vae_latent` shape `(3, 16, 9, h, w)`. Flattened timeline = 27 real latent frames. After the 19-frame zero-pad:

```
continue index:  0 ............ 18 | 19 ...... 27 | 28 ...... 36 | 37 ...... 45
content:         <-- 19 ZEROS -->  |   chunk 0    |   chunk 1    |   chunk 2
                                       (9 frames)     (9 frames)     (9 frames)
```

`choice_idx` is uniform over {0, 1, 2}. The three possible samples this clip can produce:

| choice_idx | start | history = continue[start : start+19]            | target = chunk |
|---|---|---|---|
| 0 | 0  | **19 zeros** (no past)                           | chunk 0 |
| 1 | 9  | 10 zeros + chunk 0 (9)                           | chunk 1 |
| 2 | 18 | 1 zero + chunk 0 (9) + chunk 1 (9)              | chunk 2 |

So a 3-chunk clip can train the model on **"generation start"** (k=0), **"early generation"** (k=1), and
**"a bit of history"** (k=2). It can *never* produce a fully-real 19-frame history (that needs k≥3, i.e. ≥4
chunks). That is fine and intended — these partial-history cases teach the cold-start behavior. Across epochs
the random `choice_idx` (seed = `base_seed + epoch*1e6 + idx`, dataloader:169) cycles through all three.

## 6. Worked example B — a **261-frame clip (N = 7 chunks)**

`vae_latent` shape `(7, 16, 9, h, w)`. Timeline = 63 real latent frames + 19 zeros prepended:

```
continue: [19 ZEROS] | c0(19..27) | c1(28..36) | c2(37..45) | c3(46..54) | c4(55..63) | c5(64..72) | c6(73..81)
```

`choice_idx` uniform over {0..6}. Examples:

| choice_idx | start | history (19 frames before target)                 | target | history kind |
|---|---|---|---|---|
| 0 | 0  | 19 zeros                                          | chunk 0 | all-zero (cold start) |
| 2 | 18 | 1 zero + c0 + c1                                  | chunk 2 | partial |
| 3 | 27 | last frame of c0 + c1 + c2  (19 real)            | chunk 3 | **fully real** |
| 6 | 54 | last frame of c3 + c4 + c5  (19 real)            | chunk 6 | **fully real**, deep in clip |

So a 7-chunk clip yields a much richer mix: cold-start (k=0,1,2 partial) **and** several "deep into the video,
fully-real 19-frame memory" cases (k=3..6). This is why longer clips are more valuable — more distinct
(history, target) pairs, and more *fully-conditioned* targets, from the same file. Each epoch samples a
different `choice_idx` per clip, so over training the model sees all of them.

## 7. Why `(28-1)*4+1` and not `28*4`? (the causal VAE)

The Wan VAE is a **causal temporal VAE** — temporal compression is **asymmetric**:
- the **1st** latent frame encodes only **1** RGB frame (the start keyframe, 1:1);
- **every subsequent** latent frame encodes **4** RGB frames.

So `L` latent frames ↔ RGB = `1 + (L-1)*4 = (L-1)*4 + 1`. The `-1` peels off the single 1:1 start frame; the
remaining `L-1` frames are 4:1. Check: 1 chunk = 9 latent → `(9-1)*4+1 = 33` ✓; 28 latent → `(28-1)*4+1 = 109` ✓.
You **cannot** write `28*4=112` — that assumes every latent frame is 4:1 and ignores the first frame.

## 8. "19 history latents" vs "the short stream is 2 frames" — don't conflate them

`history_sizes = [16, 2, 1]` sums to **19** — the contiguous history latent count the dataloader hands over, and
it is **always 19**. "short is 2 frames" is a *different* thing: the input **stream** fed to the `patch_short`
Conv3d has 2 frames, because an extra **x0 anchor frame** is prepended, and **x0 is not counted in the 19**. See
`utils_helios_base.py:647-659`:

```python
indices = arange(0, sum([1, 16, 2, 1, 9]))          # = arange(0, 29):  x0(1) + 16 + 2 + 1 + 9
indices_prefix, long, mid, idx_1x, hidden = split([1, 16, 2, 1, 9])
indices_latents_history_short = cat([indices_prefix, idx_1x])   # x0 + newest 1 frame = 2 frames
latents_history_long, latents_history_mid, latents_history_1x = history_latents.split([16, 2, 1])  # still 19
```

- The 19 history frames split by recency into **long=16 / mid=2 / 1x=1** (oldest 16, middle 2, newest 1).
- `patch_short` actually consumes **x0 (1 frame) + the newest 1x (1 frame) = 2 frames**. That "2" is **1 history
  frame + 1 x0 anchor**, NOT "short took 2 history frames".
- Tally: `x0(1) + history 19 + target 9 = 29` latent slots; but the **new** latent frames belonging to this window
  are still `19+9=28` (→109 RGB). **x0 is the clip's own first frame *reused* as an anchor — not double-counted.**

So `16+2+1+9 = 28` (✓) and "short is 2 frames" (`x0 + 1x`) are consistent, not contradictory.

### 8.1 The x0 anchor: why the first frame goes into the short stream (attention sink / Easy Anti-Drifting)

In autoregressive chunk-by-chunk generation, error accumulates over time → **drift** (later frames wander off the
opening, get blurry). Helios fixes the clip's **first latent frame (x0)** permanently at **position 0** of the
short stream as an immovable **attention sink / anchor**: no matter which chunk is being generated, attention always
has an "origin reference" pulling later frames back toward the opening's global tone / subject / composition.

- With `is_keep_x0=True` the dataloader returns x0 separately (the clip's true first latent frame).
- **Cold-start special case:** when `choice_idx==0` (target is chunk 0, no past yet) x0 is **zeroed** — generation
  hasn't started, there is no real anchor, so a zero teaches the model to start from scratch. For `choice_idx>0`,
  x0 is the real first frame.
- The `is_random_drop` t2v branch (`utils_helios_base.py:662-671`) zeros x0 together with all history, simulating
  pure text-to-video (no image/video condition).

## 9. How history is "compressed": the multi-term patchify token budget (at 368×640)

The 19 and 9 in §2 are **latent frame counts**; what actually enters attention are the **tokens** after patchify.
Each history granularity uses its own Conv3d (`transformer_helios.py:1066-1068`); the stride sets the compression
ratio; the target uses the main `patch_embedding` (stride `(1,2,2)`). 368×640 → latent space **46×80** (mid/long
get `pad_for_3d_conv` to round 46→48 before dividing):

| stream | input latent frames | Conv3d stride (t,h,w) | tokens = ⌈t/st⌉·⌈h/sh⌉·⌈w/sw⌉ | note |
|---|---|---|---|---|
| **long (far)** | 16 | (4, 8, 8) | (16/4)·(48/8)·(80/8) = 4·6·10 = **240** | crushed hardest: 16 frames→240 |
| **mid** | 2 | (2, 4, 4) | (2/2)·(48/4)·(80/4) = 1·12·20 = **240** | medium |
| **short (near)** | 2 (x0+newest) | (1, 2, 2) | (2/1)·(46/2)·(80/2) = 2·23·40 = **1840** | finest, barely compresses time |
| **target (to gen)** | 9 | (1, 2, 2) | 9·23·40 = **8280** | main patch_embedding |

Intuition: the **oldest 16 frames** occupy only 240 tokens while the **2 most-recent** occupy 1840 — a ~**61×**
difference in per-frame density. That is the heart of "Multi-Term Memory": **remember the far past coarsely, the
near past in detail**, keeping long-range context and recent detail under a fixed budget. The dataloader only hands
over the contiguous 19 frames (+ x0); the [16,2,1] split and three-tier patchify happen inside the model.

## 10. Worked example C — a **161-frame clip (N = 4 chunks)**, full pipeline with x0 and 3-tier compression

`floor(161/33) = 4` → `vae_latent` shape `(4, 16, 9, 46, 80)`. Flattened = 36 real latent frames; prepend 19 zeros
→ `continue` length `19+36 = 55`:

```
continue index: 0 ........ 18 | 19..27 | 28..36 | 37..45 | 46..54
content:        <- 19 ZEROS -> | chunk0 | chunk1 | chunk2 | chunk3
                                 (9)      (9)      (9)      (9)
x0 = chunk0's first latent frame = continue index 19
```

`choice_idx` uniform over {0,1,2,3}, four samples:

| choice_idx | start | history = continue[start:start+19] | target | x0 | history kind |
|---|---|---|---|---|---|
| 0 | 0  | 19 zeros                          | chunk0 | **zeroed** (cold start, no anchor) | all-zero |
| 1 | 9  | 10 zeros + chunk0(9)              | chunk1 | real first frame | partial |
| 2 | 18 | 1 zero + chunk0(9) + chunk1(9)    | chunk2 | real first frame | partial |
| 3 | 27 | chunk0 tail 1 + chunk1 + chunk2   | chunk3 | real first frame | **19 fully real** |

**Take `choice_idx=3` end-to-end** (target = chunk3, all 19 history frames real):

1. **dataloader emits** (`prepare_stage1_latent`): history = `continue[27:46]` = `[chunk0[8], chunk1(28..36),
   chunk2(37..45)]` (19); target = `continue[46:55]` = chunk3 (9); x0 = chunk0's first frame (real, not zero).
2. **split into 3 tiers** (`prepare_stage1_clean_input_from_latents`, `split([16,2,1])`):
   - long = the **oldest 16** of those 19 → patch_long(4,8,8) → **240 tokens**
   - mid = the next **2** → patch_mid(2,4,4) → **240 tokens**
   - 1x = the **newest 1** → joined with x0 into the short stream (2) → patch_short(1,2,2) → **1840 tokens**
3. **target** chunk3's 9 frames → main patch_embedding → **8280 tokens**, noised; model predicts the flow.
4. **attention sequence** ≈ `[short 1840] + [mid 240] + [long 240] + [target 8280]` (each with RoPE); x0 sits at
   short-stream position 0 as the attention sink, anchoring chunk3's generation back to the opening.
5. **flow-matching loss** is computed only on the target's 8280 tokens (history/x0 are clean conditioning — not
   noised, not a prediction target).

> Contrast: at `choice_idx=0` history is all-zero and x0 is zeroed too → the pipeline degenerates to "pure
> cold-start text-to-video first chunk", teaching the model to **open a chunk from nothing**. A 4-chunk clip covers
> cold start (k=0), two partial-history cases (k=1,2), and the first "19-frame fully-real memory" case (k=3) — richer
> than a 3-chunk clip.

## 11. One-line summary

A clip is encoded into `N = floor(frames/33)` chunks of 9 latent frames. Each training step samples **one**
chunk as the 9-frame target, takes the **19** latent frames before it (zero-padded at the start) as history, and
reuses the clip's first frame x0 as an anchor → a **fixed 28-latent (≈109-RGB) window** (causal VAE:
`(28-1)*4+1`). The 19 history frames split [16,2,1] into long/mid/short, the far past crushed to 240 tokens and the
near past kept at 1840 (coarse-far / fine-near); x0 sits at short-stream position 0 to solve attention-sink/drift.
The ≥121-frame rule guarantees `N ≥ 3` chunks, which the rollout/DMD/GAN path hard-requires
(`num_rollout_sections=3`) and which gives the random target sampler a useful spread of history states. Longer clips
(e.g. 261→7 chunks) just produce more, and more fully-conditioned, samples.

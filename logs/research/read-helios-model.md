# Helios model architecture: code-grounded report

## Scope and terminology

**OBSERVED.** This report is pinned to `/mnt/beegfs/siyuan/workspace/helios-team` on branch `mid_training_xiangbo` and deep-reads the complete 1,916-line model implementation at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py`. The model called `HeliosAttnProcessor2_0` in older descriptions is now only a deprecated constructor that returns `HeliosAttnProcessor`; the active implementation is therefore `HeliosAttnProcessor.__call__` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:172-207`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:448-455`).

A naming caveat is important: the model parameter is named `hidden_states`, but at the public `HeliosTransformer3DModel.forward` boundary it is the current noisy latent chunk, not a precomputed transformer hidden state (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1270-1289`). Below I call it **noisy/current** or `X_Noisy`; I call the concatenation of long/mid/short patchified latent history **history**.

## 1. `HeliosTransformer3DModel.forward`: signature, data flow, layout, and 480p token counts

### Full signature

**OBSERVED.** The complete signature is:

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    indices_hidden_states=None,
    indices_latents_history_short=None,
    indices_latents_history_mid=None,
    indices_latents_history_long=None,
    latents_history_short=None,
    latents_history_mid=None,
    latents_history_long=None,
    is_first_denoising_step: bool = False,
    gan_mode: bool = False,
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
```

This appears at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1270-1289`. The seven positional-index/history arguments must be either all present or all absent; a mixed state fails the assertion at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1290-1306`. One-dimensional position arrays are batch-expanded at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1308-1315`.

### Input construction before the model

**OBSERVED.** Training divides the 19-frame history according to sorted `history_sizes=[16,2,1]`, then prepends the first-frame latent `x0` to the one-frame newest tier, so the tensors passed to the model are long=16 frames, mid=2 frames, and short=`[x0,newest]`=2 frames (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:627-648`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:655-667`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:727-740`). Only the current target is flow-noised as `(1-sigma)*x + sigma*noise`; history remains a separate input, apart from optional history-corruption/saturation augmentations (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:744-757`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:826-863`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:881-888`).

### Patchification and exact sequence order

**OBSERVED.** The model defines four Conv3d embedders: current/noisy `patch_embedding` uses configurable `patch_size`, default `(1,2,2)`; the multi-term history embedders are short `(1,2,2)`, mid `(2,4,4)`, and long `(4,8,8)`, with equal kernels and strides (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:982-1009`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1018-1024`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1062-1069`). Mid and long latents are replicate-padded to the respective kernel multiples before convolution (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:50-56`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1221-1226`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1241-1246`).

`process_input_hidden_states` first patchifies and flattens current/noisy tokens (`B,C,T,H,W -> B,T*H*W,C`) at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1180-1201`. It then **prepends** short, then prepends mid, then prepends long (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1203-1220`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1221-1239`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1241-1259`). Therefore the final non-NAViT token layout is exactly:

```text
[ long-history | mid-history | short-history(x0 first, newest frame second) | current/noisy ]
```

The same prepend order is used for RoPE vectors, so token and position layouts remain aligned (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1210-1219`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1228-1239`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1248-1259`). The model retains only aggregate `original_context_length` and `history_context_length`, not per-tier history lengths, after patchification (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1330-1351`).

### Typical 480p count

**OBSERVED.** I assume the code's validation default of **480x832 RGB** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/train_config.py:105-110`), Wan VAE spatial scale 8 and temporal scale 4 (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:155-167`), and one standard 9-latent-frame chunk (`history_sizes=[16,2,1]`, `latent_window_size=9`: `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:901-906`). Thus the latent grid is `T=9,H=60,W=104`; pipeline latent allocation divides RGB height/width by the VAE factor (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:348-355`). Counts are:

| token group | latent input | post-patch grid | tokens |
|---|---:|---:|---:|
| long history | `16 x 60 x 104` | `4 x ceil(60/8) x 13 = 4 x 8 x 13` | **416** |
| mid history | `2 x 60 x 104` | `1 x 15 x 26` | **390** |
| short history (`x0` + newest) | `2 x 60 x 104` | `2 x 30 x 52` | **3,120** |
| current/noisy | `9 x 60 x 104` | `9 x 30 x 52` | **14,040** |
| all history | — | — | **3,926** |
| total | — | — | **17,966** |

The count follows directly from the Conv3d kernels/strides and mid/long padding cited above. **ABSENT.** No constant “480p token count” is stored in the model; it is shape-derived at runtime.

### Remaining forward flow

**OBSERVED.** `forward` obtains patch tokens/RoPE and length metadata through `process_input_hidden_states` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1330-1346`), constructs timestep and text conditioning (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1353-1366`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1433-1449`), then sends the concatenated sequence through every block (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1451-1486`). Output processing discards history, normalizes and projects only the last `original_context_length` current/noisy tokens, and unpatchifies them to latent video (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1488-1555`).

## 2. Attention structure, text cross-attention, and `guidance_cross_attn`

### Is `[noisy,history]` one fully bidirectional self-attention?

**OBSERVED.** It depends on `restrict_self_attn`:

1. **`restrict_self_attn=False`: yes.** Q/K/V are projected from the entire concatenated sequence and passed to a single unmasked attention call (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:227-230`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:320-326`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:410-415`). With no mask, the dispatched fallback is ordinary scaled-dot-product attention and has no causal flag (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/helios_kernels/attention_dispatch.py:132-149`). Therefore all concatenated history and current/noisy queries can read all concatenated keys bidirectionally.
2. **`restrict_self_attn=True`: no; it is asymmetric.** The processor splits the prefix history from the suffix current sequence at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:301-304`. History Q attends only history K/V at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:372-379`; current Q attends concatenated `[K_history,K_current]` and `[V_history,V_current]`, whose exact concatenation is `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:372-373`, followed by attention at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:410-415`. History outputs and current outputs are re-concatenated at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:419-441`. There is no causal mask inside either group.

**OBSERVED.** The released training configs shown in this branch set `restrict_self_attn: false`, e.g. `/mnt/beegfs/siyuan/workspace/helios-team/scripts/training/configs/stage_1_init.yaml:177-179`; the explicit cache path requires restricted attention, as validated by `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:2717-2718`. Thus one must not describe *all* Helios operation as asymmetric: unrestricted joint attention is the configured training path, while restricted asymmetric attention is the cache-compatible optional path.

### Cross-attention to text

**OBSERVED.** Every block owns a separate `attn2` cross-attention module whose query comes from video tokens and whose K/V come from `encoder_hidden_states` because `_get_qkv_projections` uses `hidden_states` for Q and encoder states for K/V in cross-attention (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:75-92`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:755-765`). UMT5-like input text embeddings are first projected by `PixArtAlphaTextProjection` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:623-638`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:663-666`). This is independent of self-attention and does not use video RoPE (`rotary_emb=None` in the `attn2` calls: `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:871-879`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:883-892`).

### What `guidance_cross_attn` does

**OBSERVED.** `guidance_cross_attn=True` does **not** create a separate guidance input and is not classifier-free-guidance arithmetic. It prevents history tokens from receiving text cross-attention. In the non-NAViT path it splits history/current, runs `attn2` only on current, adds the result, then restores the unchanged history prefix (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:823-825`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:865-881`). In NAViT it performs the equivalent per-sequence extraction, cross-attention over concatenated current tokens, and reinsertion of each history prefix (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:826-864`). When false, all video tokens, including history, cross-attend to text (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:882-893`). NAViT cross-attention mask query lengths likewise exclude history only when this flag is true (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/helios_kernels/attention_dispatch.py:94-117`).

## 3. `is_amplify_history`: exact mechanism, parameters, initialization, persistence

**OBSERVED.** `is_amplify_history` is threaded model -> block -> self-attention at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1008-1009`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1033-1051`, and `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:738-753`. Each block's `attn1` owns either a scalar `history_key_scale` or a length-`heads` parameter, initialized to **ones**, plus `max_scale=10` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:518-528`). Per-head is the default (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:477-478`).

The effective multiplier is not the raw initialized value. It is

```text
scale = 1 + sigmoid(history_key_scale) * (10 - 1)
```

at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:530-537`; with raw init `1`, the initial effective scale is approximately **7.5795**, not 1. Per-head scaling is reshaped to `[1,1,H,1]` and multiplies only the history-prefix **keys**, never values or queries (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:394-408`). In non-NAViT it assumes history occupies the first `history_seq_len` positions in the final K sequence (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:408-408`).

**OBSERVED.** The parameter is an ordinary `nn.Parameter`, and the class marks names containing `history_key_scale` to remain FP32 during custom loading (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:944-960`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1780-1786`). The trainer explicitly makes these parameters trainable when enabled (`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:463-477`). The special partial-checkpoint path writes one scale tensor per block to `transformer_partial.pth` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_base.py:223-256`) and restores it when `args.training_config.is_amplify_history` is enabled (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_base.py:351-363`). Standard complete `state_dict` save/load also naturally includes an `nn.Parameter`; the custom loader instantiates from config, filters only name/shape-compatible tensors, and calls `load_state_dict(strict=False)` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1745-1777`).

**ABSENT.** There is no separate scale per short/mid/long tier, no scale for history values, and no learned memory-group scale.

## 4. `HeliosRotaryPosEmbed`: IDs, tiers, anchor, stationarity

### Position construction

**OBSERVED.** RoPE divides each 128-dimensional head into temporal/spatial axes `rope_dim=(44,42,42)`, uses theta 10,000, and constructs cosine/sine frequencies for temporal frame ID plus post-patch `y,x` grids (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:982-1000`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:669-685`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:696-714`). Current/noisy tokens use `indices_hidden_states`, while each history tier uses its corresponding index tensor (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1183-1196`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1210-1216`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1228-1236`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1248-1256`). Mid and long RoPE grids are computed at short-tier spatial resolution, replicate-padded, then average-pooled by `(2,2,2)` or `(4,4,4)` to match coarse tokens (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1228-1236`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1248-1256`).

### Exact default temporal IDs

**OBSERVED.** With `history_sizes=[16,2,1]`, `latent_window_size=9`, and anchor retention, input preparation makes `arange(0,29)` and splits it as:

```text
anchor/prefix: 0
long history:  1..16
mid history:   17..18
newest 1x:     19
current/noisy: 20..28
short tier:    [anchor 0, newest 19]
```

The construction and split are at `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:646-665`; inference repeats exactly the same construction for every section at `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1265-1273` and `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1284-1292`.

Coarse-tier temporal phases are pooled, not assigned a single integer. Given those IDs, long groups `[1..4],[5..8],[9..12],[13..16]` become mean RoPE frequency vectors over each four-frame group; mid IDs `[17,18]` become one mean vector over the pair; short retains IDs `0` and `19`; current retains `20..28`. This follows from computing per-frame trigonometric frequencies first and then applying average pooling (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:681-685`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1228-1236`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1248-1256`).

### Relative/window-stationary versus absolute

**OBSERVED.** Across autoregressive sections, these IDs are reset to the same `0..28` window template; they do not grow with global video time (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1236-1298`). Meanwhile the latent content placed in long/mid/short is refreshed from the latest rolling `history_latents` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1294-1298`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1452-1454`). Therefore Helios uses **window-stationary/relative slot IDs**, not absolute global frame IDs. The fixed first-frame anchor always occupies slot 0; the rolling history is reassigned to slots 1..19; every new chunk is reassigned to 20..28.

**ABSENT.** There is no section counter added to RoPE, no global frame offset, and no dedicated learned position embedding for the anchor.

## 5. Timestep conditioning for history

**OBSERVED.** `HeliosTimeTextEmbedding` projects scalar timesteps into (a) `temb` for output AdaLN and (b) a `6*D` vector used by each transformer's six shift/scale/gate terms (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:623-666`). Each block adds its learned `scale_shift_table` to this projected timestep, applies timestep-conditioned normalization/gating around self-attention and feed-forward, and applies an un-gated residual for text cross-attention (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:771-820`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:822-900`).

When history indices are present and `zero_history_timestep=True`, `forward` computes a separate embedding for exactly `timestep=0`, expands it to `history_context_length`, and prepends it to the current tokens' real-timestep embeddings (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1353-1366`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1433-1447`). Thus all long/mid/short tokens, including the anchor, receive t=0 AdaLN conditioning in every block. Current/noisy tokens receive the requested denoising timestep. The output path trims to current tokens before output normalization/projection (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1547-1555`).

If `zero_history_timestep=False`, the current timestep is expanded over the whole concatenated sequence (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1433-1443`). **ABSENT.** There is no tier-specific history timestep embedding and no special anchor timestep distinct from the other history tokens.

## 6. Dormant/optional KV-cache infrastructure

**OBSERVED.** Each self-attention processor owns an in-memory Python `kv_cache`, an enable flag, and enable/disable/clear methods (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:172-194`); model-wide wrappers invoke these on every block's `attn1` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1118-1131`). It is specifically built to avoid recomputing **fixed history** during iterative denoising when `restrict_self_attn=True`: on the first denoising step it caches `key_history`, `value_history`, and the history attention output (`history_hidden_states`) (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:375-388`); on later steps it removes history input tokens/RoPE, projects only current tokens, prepends cached K/V to current K/V, and restores cached history hidden output (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:208-225`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:328-331`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:419-441`). Cache and NAViT are explicitly incompatible (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:215-217`).

**OBSERVED.** It is not literally unreachable dead code. The pipeline exposes `use_kv_cache=False`, enables it when requested, passes `is_first_denoising_step`, and clears it after each generated section (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:928-929`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1000-1002`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:517-540`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1429-1430`). However, it is optional and off by default both in pipeline invocation and validation config (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/train_config.py:105-126`). It is also **section-local**, deliberately cleared once a section finishes; it is not the autoregressive history store. Persistent history remains latent frames appended to `history_latents` and re-patchified next section (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1452-1454`).

**INFERRED (design implication).** Its per-block storage could be extended to hold persistent memory K/V, but it cannot host Echo-style evolving state unchanged: it currently owns only one unnamed history prefix, assumes cache validity only across denoising steps of the same section, clears at the section boundary, stores no memory-query hidden state/gate metadata/RoPE IDs, and cannot operate with NAViT. A robust port should use an explicit structured memory state (or substantially redesign this cache) rather than silently overloading `key_history`.

## 7. First-frame anchor (“sink”)

**OBSERVED.** The first-frame anchor is a **latent frame**, not a learned sink token and not persistent KV. In training, `x0_latent` is copied from the source video's first latent frame; for the very first target section it is zeroed (`/mnt/beegfs/siyuan/workspace/helios-team/helios/dataset/dataloader_history_latents_dist.py:136-145`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/dataset/dataloader_history_latents_dist.py:169-175`). `prepare_stage1_clean_input_from_latents` concatenates it with the newest one-frame history to form the two-frame **short** history tier (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:641-665`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:727-740`). It is therefore **not part of the long tier**.

At inference `is_keep_x0=True` by default (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:901-906`). After the first generated section (for T2V), `image_latents` is set to the first generated latent frame; for I2V, it is the supplied image latent (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1280-1282`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1447-1450`). Every later section concatenates this same `image_latents` prefix with the newest one-frame history (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1284-1298`). Its temporal RoPE slot is always 0, while the newest short frame uses slot 19, as established in Section 4.

**ABSENT.** There is no learned attention-sink vector, no sink-only attention mask, no separate sink projection, and no anchor-specific amplification. The anchor shares short-tier patchification, t=0 conditioning, attention rules, and aggregate history amplification with all history tokens.

## 8. Hidden-state tap points and 14B dimensions

**OBSERVED.** The 14B/Wan defaults are 40 layers, 40 heads, head dimension 128, model/inner dimension `40*128=5120`, FFN dimension 13,824, latent input/output channels 16, and default patch `(1,2,2)` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:982-1019`). Self-attention confirms `dim_head=dim//num_heads` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:738-753`).

The cleanest last-block tap is immediately after the final assignment `hidden_states = block(...)` and before output normalization, i.e. at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1473-1486` followed by the output boundary at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1488-1500`. In non-NAViT 480x832 operation its shape is **`[B, 17,966, 5,120]`**, laid out `[long 416 | mid 390 | short 3,120 | current 14,040]`. Current/noisy only is `hidden_states[:, -original_context_length:, :]`, shape **`[B,14,040,5,120]`**, exactly the slicing pattern already used for GAN hooks (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1470-1471`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1485-1486`) and the non-NAViT output (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1547-1550`).

**OBSERVED.** A tap “before `proj_out`” can mean two different tensors: (a) raw last-block hidden states before output AdaLN, at lines 1473-1488; or (b) output-normalized current tokens immediately before the linear projection, produced at lines 1548-1550 (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1547-1551`). For Echo's “last-block hidden states,” (a) preserves the actual block output and is the semantically direct tap.

**INFERRED (implementation caution).** Returning the full tensor at 480p costs about 184 MB per sample in bf16 (`17,966*5,120*2` bytes), before autograd retention. Capturing only selected current tokens or a compressed write set is materially safer.

## 9. `LoRALinearLayer` and `restrict_lora`

**OBSERVED.** `LoRALinearLayer` is a bias-free low-rank residual `down(in->rank)` then `up(rank->out)`; down weights initialize normal with std `1/rank`, up weights initialize to zero, so the initial residual is exactly zero (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:129-154`). This is separate from PEFT's general model LoRA.

When `restrict_lora=True`, every self-attention block instantiates dedicated `q_loras`, `k_loras`, and `v_loras`; their parameters are trainable only if `is_train_restrict_lora=True` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:513-516`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:539-548`). The processor adds these residuals **only to history Q/K/V after the history/current split**, both in non-NAViT and NAViT paths (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:267-289`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:301-309`). Current/noisy Q/K/V continue to use the base projections. The trainer also explicitly unfreezes these module names when requested (`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:463-477`), and partial save/load handles them at `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_base.py:201-221` and `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_base.py:297-349`.

**ABSENT.** `restrict_lora` does not impose an attention mask by itself; it only exists inside the `restrict_self_attn` splitting branch. It also does not provide a distinct adapter per short/mid/long tier.

## 10. Insertion analysis for `[noisy, history, memory]`

The requested semantic order is described as `[noisy, history, memory]` at K/V, but Helios's physical sequence is currently `[history,current/noisy]`. The safest change is to define one canonical physical layout and pass explicit lengths; otherwise “first prefix” and “last suffix” assumptions will silently misroute memory. The following are the exact change sites.

### 10.1 Public inputs, configuration, and module construction

1. **`HeliosTransformer3DModel.__init__`** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:982-1015`): add memory feature/config flags, memory amplification mode, and any projection/embedding modules. Thread them into every block where history amplification and restricted adapters are currently wired (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1033-1053`). Update `_keep_in_fp32_modules`, which currently names only `history_key_scale` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:944-960`).
2. **`HeliosTransformerBlock.__init__` / `HeliosAttention.__init__`** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:719-753`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:462-528`): add a separately owned `memory_key_scale` (and, if desired, memory-specific adapters), rather than reusing `history_key_scale`. Current code has one scale and one `history_scale_mode` only.
3. **`HeliosTransformer3DModel.forward`** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1270-1306`): accept memory tokens/state and memory RoPE IDs. Replace the all-or-none assertion, which enumerates exactly seven current/history fields and knows no memory group. Define behavior when history is absent but memory is present, and vice versa.

### 10.2 Tokenization, RoPE, and length bookkeeping

4. **`process_input_hidden_states`** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1133-1268`): insert memory tokens and their RoPE in the canonical order. It currently handles only current plus three latent-history tensors, prepending short/mid/long and returning no group-specific lengths. Return at least `history_context_length` and `memory_context_length` explicitly; do not infer both from one residual.
5. **RoPE generation** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:669-714` plus caller sites `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1183-1259`): `HeliosRotaryPosEmbed` itself already accepts arbitrary frame IDs, so its frequency math need not change if memory has a 3D grid. The callers must provide memory slot IDs and shape-aligned spatial IDs. For unstructured learned queries, add a well-defined spatial convention because the current API requires `height,width` and emits exactly `T*H*W` vectors. The current fixed temporal slots 0..28 are fully occupied; adding memory between anchor and long history requires rebuilding the index generator at `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:655-665` and inference generators at `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1265-1292`.
6. **Forward length computation** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1347-1351`): `history_context_length = total-current` currently collapses every non-current token into history. This must become explicit three-group bookkeeping. All downstream arguments named `history_context_length` need audited semantics.

### 10.3 Timestep/AdaLN conditioning

7. **t=0 embedding construction and concatenation** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1353-1366`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1419-1447`): expand t=0 over `history_len + memory_len` if both are clean, or build separate spans if memory later needs its own conditioning. Current code creates one t=0 prefix of `history_context_length` and concatenates it before the current timestep span; it assumes every non-current token is history and all conditioning is a two-way prefix/suffix split.
8. **Per-block AdaLN** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:792-820`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:895-900`): no formula change is required if `timestep_proj` is length-aligned, but tests must verify the memory span receives t=0 in all six modulation tensors.

### 10.4 Self-attention routing and K/V concatenation

9. **`HeliosAttention.forward` / processor argument contract** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:599-620`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:196-207`): pass explicit group lengths or structured slices. The current processor derives `history_seq_len` by `(total-current)/num_sequences` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:208-214`), which cannot distinguish memory.
10. **Non-NAViT hardcoded two-way split** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:301-319`): replace `[:history_seq_len]` / `[history_seq_len:]` for hidden/Q/K/V/RoPE with explicit history, memory, and current slices. Decide memory-query semantics: if memory is KV-only, do not produce/reinsert memory Q output; if it evolves through self-attention, define what its Q can see.
11. **NAViT hardcoded two-way split and reversal logic** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:232-300`): every per-sequence `seq_end`, prefix slice, concatenation, and cursor increment assumes `history_len+current_len`. Add memory length and preserve each group's identity.
12. **Exact K/V assembly** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:328-379`): current non-NAViT current-query keys are exactly `[K_history,K_current]`. Change this to the selected canonical order, e.g. `[K_history,K_memory,K_current]` or the requested `[K_current,K_history,K_memory]`; values must match. NAViT assembly at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:333-370` must change in parallel. Since softmax attention is permutation-equivariant over aligned K/V pairs, order does not alter math by itself, but every prefix scale/cache/mask assumption makes consistent ordering essential.
13. **Output reassembly** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:419-441`): currently concatenates exactly history output then current output. Reinsert (or deliberately omit) memory-query outputs consistently.

### 10.5 Independent amplification

14. **Scale application** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:394-408`): it currently scales the first `history_seq_len` K positions and treats everything else as current. Apply `history_key_scale` and `memory_key_scale` to explicit, non-overlapping slices. In NAViT update the per-sequence cursor from `history+current` to `history+memory+current`. Preserve the fact that amplification is key-only unless the design explicitly changes that invariant.
15. **Persistence and training plumbing**: extend the trainer's model kwargs/trainable-module list (`/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:310-320`, `/mnt/beegfs/siyuan/workspace/helios-team/train_helios.py:463-477`) and partial save/load, which currently extracts only `history_key_scale` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_base.py:223-236`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_base.py:351-363`). Add backward-compatible missing-key initialization for existing checkpoints.

### 10.6 Cross-attention guidance and masks

16. **`guidance_cross_attn` split/reassembly** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:823-881`): it currently excludes one prefix (`history`) from text cross-attention. Decide explicitly whether memory queries should cross-attend to text; then split `[history,memory,current]` instead of treating memory as history accidentally. Both NAViT and non-NAViT branches are hardcoded two-way.
17. **NAViT masks** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/helios_kernels/attention_dispatch.py:46-119` and call `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1368-1383`): all Q/KV cumulative lengths are formulas over `current + history`; restricted history attention has a separate history-only mask. Add memory to the applicable KV lengths and, if memory has Q attention, create the required memory mask. Cross-attention query lengths must reflect whether memory is text-guided.

### 10.7 Cache semantics

18. **Processor cache** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:182-225`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:383-388`): it stores only history K/V/output under fixed field names and strips exactly one prefix on reuse. Add separate memory K/V and any evolving hidden state, include validity/version metadata, and define detach/update behavior across sections. If memory evolves within denoising, it cannot be cached like immutable history.
19. **Pipeline lifecycle** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1000-1002`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1429-1454`): current cache is cleared at every section while latent history persists. An evolving memory state needs an explicit section-to-section owner and update after the chosen final/near-zero denoising step. Do not rely on the current cache clear behavior.

### 10.8 Last-state capture and output

20. **Last-block capture** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1451-1488`): expose selected raw final-block current hidden states before output AdaLN/projection. Existing `Transformer2DModelOutput` returns only `sample` and GAN `logits` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1557-1578`), so add a backward-compatible optional return field or callback. Avoid overloading GAN hooks, which slice current tokens but are conditional on GAN mode.
21. **Output trimming/unpatchify** (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1508-1555`): non-NAViT's suffix slice remains correct only if current/noisy stays last. NAViT assumes each sequence is `[history,current]` and slices `history_context_length`; update each per-sequence span and current offset for memory (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1511-1529`).

### Critical absent pieces

**ABSENT.** The current model contains no memory-query parameter bank, no write gate, no memory encoder/update rule, no memory-slot RoPE convention, no persistent section-level memory state, no explicit third-group sequence lengths, no memory-specific amplification, and no API to return last-block current hidden states. Those are new architecture, not latent features waiting to be enabled.

## TL;DR

- Helios physically tokenizes `[long | mid | short(x0,newest) | noisy]`; at 480x832 RGB / 9 latent frames this is `416 + 390 + 3,120 + 14,040 = 17,966` tokens (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1180-1259`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/train_config.py:105-110`).
- Unrestricted self-attention is one fully bidirectional joint attention over history and noisy tokens; restricted mode instead makes history read-only and current queries read `[history,current]` K/V (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:227-230`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:301-379`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:410-441`).
- K/V are concatenated at `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:372-373`; `HeliosAttnProcessor2_0` is only a deprecated constructor for `HeliosAttnProcessor` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:448-455`).
- `guidance_cross_attn` means text cross-attention updates current tokens only and leaves history unchanged; it is not CFG arithmetic (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:823-893`).
- Every block can learn scalar/per-head history **key-only** amplification; raw init is 1, yielding an effective initial multiplier `1+sigmoid(1)*9 ~= 7.58` (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:518-537`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:394-408`).
- RoPE IDs are window-stationary: anchor 0, long 1..16, mid 17..18, newest short 19, noisy 20..28; inference rebuilds these same slots every section (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:655-665`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1265-1292`).
- With `zero_history_timestep=True`, all history/anchor tokens get t=0 AdaLN conditioning, while noisy tokens get the current denoising timestep (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1353-1366`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1433-1447`).
- The KV cache is optional/off by default, restricted-attention-only, denoising-step-local, NAViT-incompatible, and cleared after each section; persistent history is latent frames, not KV (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:208-225`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:383-388`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:928-929`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/pipelines/pipeline_helios.py:1429-1454`).
- The first-frame anchor is a fixed latent frame in the short tier, not a learned sink token and not part of long history (`/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:641-667`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/utils/utils_helios_base.py:727-740`).
- The direct Echo write tap is raw final-block current hidden state before output AdaLN/projection: `[B,14,040,5,120]` for the assumed 480p run (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1473-1488`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:982-1019`).
- `restrict_lora` adds separate zero-initialized low-rank Q/K/V residuals only to history projections (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:129-154`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:301-309`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:539-548`).
- A third memory group requires explicit three-way lengths/slices across tokenization, RoPE, AdaLN, K/V assembly, amplification, guidance cross-attention, NAViT masks, cache lifecycle, checkpoint plumbing, and output trimming; the current public inputs and two-way length derivation contain no such group (`/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:208-214`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1270-1306`, `/mnt/beegfs/siyuan/workspace/helios-team/helios/modules/transformer_helios.py:1347-1351`).

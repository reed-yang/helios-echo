# Echo-Infinity Learnable Evolving Memory: Code-Level Report

## Scope and interpretation

This report treats the requested “three-tier KV” mechanism as **(1) a fixed sink prefix, (2) learnable/evolving query-memory KV, and (3) a rolling recent/local KV window**. The principal deployed path is the `CausalWanModelInfinityMemory` specialization selected by the wrapper when both infinite attention and relative RoPE are enabled (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/wan_wrapper.py:108-118`). A second, more feature-rich implementation of query memory plus an experimental evolving `SinkMemory` is also embedded in `causal_model.py` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:547-580`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:726-739`). I distinguish these paths explicitly.

**Critical absence:** the specialized infinity-memory class is inference-only: its attention forward raises without `kv_cache`, so it has no native no-cache/training attention path (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:23-25`). Long-memory training nevertheless exercises inference-style cached forwards through the streaming pipeline, and the trainer installs the encoder outside FSDP on the inner model (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:204-216`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:237-251`).

## 1. Three-tier KV cache structure

### 1.1 Constants in the shipped long-memory recipes

The long recipe fixes `num_frame_per_block=3`, `local_attn_size=12`, `sink_size=3`, and relative-RoPE `pmax=21` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:55-62`). It fixes query memory to `Q_frames=3`, `tokens_per_frame=1560`, hidden width 1536, 12 heads, and head width 128 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:106-114`). The long inference recipe repeats the same 3/12/3/21 and 3×1560 query-memory settings (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long_inference.yaml:8-15`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long_inference.yaml:31-39`).

`frame_seq_length` is hard-coded as 1560 in the inference pipeline, which allocates 30 transformer-layer caches (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:28-30`). Each per-layer cache contains `k,v` of shape **[batch, kv_cache_size, 12, 128]**, in the caller-provided `dtype`, plus scalar-long `global_end_index` and `local_end_index` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:133-143`). With `local_attn_size=12`, the pipeline allocates `kv_cache_size=12×1560=18,720` tokens (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:60-69`).

The self-attention constructor likewise translates a finite local setting into `max_attention_size = local_attn_size × 1560` tokens (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity.py:43-49`). Importantly, the 12-frame cache/attention capacity **includes the sink prefix**: the active recent budget is calculated as `max_attention_size - sink_tokens` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:266-270`). Therefore, with the shipped settings:

- sink tier: **3 frames = 4,680 tokens**;
- query-memory tier: **3 frame-equivalents = 4,680 tokens**;
- recent/local tier at steady state: **12 − 3 = 9 frames = 14,040 tokens**;
- attention KV presented to a normal 3-frame generation block after memory becomes active: **4,680 + 4,680 + 14,040 = 23,400 tokens**, ordered sink → memory → local (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:109-120`).

The memory tier does **not** occupy storage in the rolling `kv_cache`; it is generated separately by `QueryMemoryEncoder.get_kv()` and only concatenated for attention (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:194-212`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:102-120`).

### 1.2 Exact tensor shapes and dtypes

Let batch size be `B`, memory token count `M=Q_frames×tokens_per_frame=3×1560=4680`, heads `H=12`, head width `D_h=128`, and hidden width `D=1536` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:40-55`).

- Persistent query state: **[B, M, D] = [B, 4680, 1536]** (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:102-117`).
- Encoder update source `evicted_k, evicted_v`: **[B, E, H, D_h]**, where `E=num_exited_frames×1560`; capture is sliced from the last layer’s per-layer cache (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:277-288`).
- Memory output `k,v`: **[B, M, H, D_h] = [B, 4680, 12, 128]** (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:194-212`).
- Each layer’s sink/local cache: **[B, 18720, 12, 128]** under the long recipe (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:60-69`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:141-143`). Its logical layout is `[sink prefix | rolling local tokens | unused tail]`, with `local_end_index` delimiting live tokens; rolling explicitly preserves `[0:sink_tokens]` and shifts only the post-sink region (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:34-49`).
- Attention Q/K/V internally use **[B, tokens, H, D_h]**, and outputs flatten heads back to `[B,tokens,D]` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:13-23`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:132-135`).

The rolling cache dtype is the caller’s noise dtype; inference samples noise as BF16 and passes that dtype to cache allocation (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/inference/inference.py:129-133`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/inference/inference.py:167-168`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:69-70`). The memory encoder is reset to BF16 explicitly in the specialized inference path (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:212-218`), and training constructs it in BF16 when mixed precision is enabled (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:209-216`). Attention normalizes runtime conversion: the FlashAttention wrapper converts non-half tensors to its configured BF16, then casts Q/K to V dtype, and returns the original Q dtype (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/attention.py:21-28`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/attention.py:41-52`). The SDPA fallback also casts Q/K/V to BF16 by default (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/attention.py:54-66`).

## 2. Memory-query parameterization

`QueryMemoryEncoder` owns the learnable initialization directly. For the normal one-group configuration, `query_init` is an `nn.Parameter` of shape **[1,M,D]**, initialized from `Normal(0,1)×initializer_range`; the recipe sets `initializer_range=0.014` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:67-68`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:82-91`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:108-114`). Reset expands this one learned template across batch, clones it, and moves/casts it to requested device/dtype (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:102-117`).

The shipped configs do not set `num_query_groups`, so the default is one (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:65-67`). Consequently the evolving hidden query state and its final K/V projections are **shared across all DiT layers**: the specialized model calls `enc.get_kv()` once and repeats the same tuple for every block (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:219-231`). Optional grouped mode exists: it creates one learned `query_init`, one K projection, one V projection, one connector, and one gate per group (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:71-81`), then maps contiguous blocks to group KVs (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:221-229`). **Absent in all root `configs/*.yaml`: any `num_query_groups` override, so deployed root recipes remain layer-shared.**

The encoder is deliberately kept outside FSDP: reset emits an error explaining that FSDP flattening is unsupported (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:104-107`), and the trainer attaches the separately constructed encoder with `object.__setattr__` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/trainer/distillation.py:237-251`).

## 3. Eviction

### Trigger and frames that leave

At each layer, rolling is triggered only when all three conditions hold: finite local attention, the call advances beyond the previous global end, and `num_new_tokens + local_end_index > kv_cache_size` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:31-35`). The excess is `num_evicted_tokens`; surviving post-sink tokens are shifted left while the sink prefix is untouched (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:35-48`). Thus eviction removes the **oldest non-sink local frames**. With aligned 3-frame blocks and 1560 tokens/frame, steady-state eviction is three frames per committed context call.

Model-level memory writing independently computes the chronological local-window boundary. It derives recent capacity as `(max_attn − sink_tokens)/1560`, computes `oldest_recent_frame=max(sink_frames,current_end_frame−recent_window_frames)`, compares it to `_ei_prev_window_start`, and defines the number leaving as their positive difference (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:266-278`). This means no write occurs until the recent tier actually advances past an old frame.

### Which layer is the write source

The “last layer” claim is **verified for query memory**. Cache-update records are appended in transformer-block iteration order (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:241-260`). The code takes `last_block_idx = cache_update_infos[-1][0]`, slices exited K/V from `kv_cache[last_block_idx]`, and sends only those tensors to `query_memory_encoder.update()` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:261-288`). The model has 30 cache blocks in the pipeline (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:28-30`), so this is block index 29 in the standard complete forward.

**Nuance:** this is last-layer **K/V**, not last-layer hidden states. K is post-`self.k` and `norm_k`, V is post-`self.v` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:18-23`). Under the specialized relative-RoPE path they are cached pre-RoPE and rotated only at read time (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:80-99`).

The experimental `SinkMemory` differs: it captures exited K/V from **every** block into `evicted_kv_all` before invoking `sm.update()` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:829-836`). Its write source is therefore layer-matched, not last-layer-only.

### Exact ordering and pre-update capture

Within one specialized forward, every block first computes against a cloned temporary cache and returns a deferred update descriptor (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:39-63`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:241-260`). After all blocks finish, the model:

1. computes exited-frame count and capture offset (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:261-283`);
2. clones exited K/V from the **current persistent, pre-roll last-layer cache** (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:283-288`);
3. updates the query state (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:288`);
4. only then applies all deferred cache rolls/inserts (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:289`; base implementation at `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity.py:345-377`).

For a roll, `capture_start=sink_tok`, exactly where the oldest non-sink items reside before shifting; otherwise it uses `prev_oldest×1560` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:277-285`). This ordering avoids accidentally encoding newly inserted K/V as evicted history.

## 4. Write path: encoder, gate, and recurrent update

### Encoder architecture

`QueryMemoryEncoder` defaults to two `MemoryCrossAttentionLayer`s, with `D=1536`, `H=12`, `D_h=128`, and FFN width `4D=6144` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:40-49`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:68-68`). Each layer contains:

- RMSNorm on query state;
- learned `q: Linear(D,D)` plus optional RMSNorm on Q;
- cross-attention using **cached K/V directly**, with no encoder-owned K/V projections;
- learned output `Linear(D,D)` and residual;
- second RMSNorm plus `Linear(D,4D) → GELU(tanh) → Linear(4D,D)` FFN and residual (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:8-34`).

With standard settings, update context is just evicted K/V; optional sink anchoring prepends sink K/V, but root recipes set `use_sink_anchor:false` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:124-142`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:117-120`). Each update feeds the recurrent prior query state through both cross-attention layers in order (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:144-147`).

After the cross-attention stack, the normal path applies a connector `Linear(D,D) → GELU(tanh) → Linear(D,D) → RMSNorm(D)` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:83-85`).

### Gate formula and initialization

The gate is elementwise over the full hidden width:

`projected = connector(state)`

`gate = sigmoid(Linear([old_query ; projected]))`

`new_query = gate * old_query + (1 - gate) * projected`

This is exactly implemented at `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:144-155`. The gate linear maps `2D→D`; its bias is filled with `gate_init_bias` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:84-87`). Recipes set the bias to **+2.0** (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:121-124`), so at initialization the bias-only retention coefficient is `sigmoid(2)≈0.8808`, with the candidate write coefficient `≈0.1192`. This is an EMA-like learned residual update, but the coefficient is token- and channel-dependent, not a scalar.

**Important naming correction:** there is no separate fixed EMA decay in this memory write. The implemented “residual EMA” is the gate equation above. Optional `use_residual_update=True` bypasses the connector/gate and assigns the cross-attention-stack output directly (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:148-155`), but this flag is absent from root configs and defaults false (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:63-64`). The unrelated trainer-level model-weight EMA is configured separately (`ema_weight`) and is not the query-memory recurrence (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:36-37`).

Optional buffered updates concatenate multiple evictions on the token axis before encoding, but root recipes set `use_batch_update:false` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:128-136`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:117-119`). After each real update, `has_history=True`, enabling injection on the **next** model forward (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:187-192`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:219-231`).

## 5. Injection into self-attention

### K/V projections and sharing

Refreshed query hidden state is converted to K/V by encoder-owned `to_k: Linear(D,H×D_h)` and `to_v: Linear(D,H×D_h)` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:88-91`). `get_kv()` reshapes these to `[B,M,H,D_h]`; optionally only K is RMS-normalized (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:194-212`). The long training recipe sets `normalize_memory_k:false`, while the short recipe sets it true (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:122-125`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity.yaml:86-88`).

These are **not** the DiT blocks’ own `self_attn.k`/`self_attn.v` weights. They are dedicated `QueryMemoryEncoder` projections, shared by all DiT layers in one-group mode (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:83-91`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:228-230`). Optional groups provide per-group projections, not per-layer projections unless configured with a group for every layer (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:71-78`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:221-227`).

### Concatenation and masking

At each layer, the specialized path rotates sink K, local K, current Q, and—when active—memory K into their assigned relative positions (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:79-108`). It concatenates exact order **`[K_sink, K_memory, K_local]`** and matching V order (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:109-120`), then calls `attention(roped_query,k_cat,v_cat)` with no lengths, no explicit mask, and default `causal=False` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:132-135`; defaults at `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/attention.py:54-56`). Thus every current query token can attend all concatenated tiers. Causality is imposed at the **block rollout level**—three frames are generated as one causal block—rather than by a triangular mask inside this inference attention call. Bulk forwards (`B > num_frame_per_block`, typically recache) explicitly disable memory injection (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:63-78`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:102-108`).

## 6. Unified bounded relative RoPE

### Position layout

The shared relative-position function computes:

- `q_last = min(current_start_frame + B - 1, pmax - 1)`;
- `q_start = q_last - B + 1`;
- local range ending at `q_last`, with `local_start = local_end - R + 1`;
- when memory is active, memory ends immediately before local: `mem_end=local_start-1`, `mem_start=mem_end-N_Q+1`;
- sink always starts at 0 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:63-78`).

With the steady-state shipped constants `pmax=21`, `B=3`, `N_S=3`, `N_Q=3`, `R=9`, long rollout saturates at:

- sink IDs **0..2**;
- memory IDs **9..11**;
- local IDs **12..20**;
- current-query IDs are the tail of local, **18..20** for the three current frames.

This follows directly from the implemented formulas and the 9-frame recent budget above. The code asserts query bounds, memory not overlapping sink, memory/local contiguity, and query/local-tail equality (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:70-78`).

A key semantic detail is that current K/V are already inserted into the temporary local cache before attention (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:39-63`); therefore Q IDs coincide with the final `B` local IDs, rather than occupying a fourth disjoint tier.

### Why IDs stay bounded indefinitely

Absolute rollout time only affects `q_last` until it reaches `pmax−1`; after that it is clamped forever (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:63-68`). Local and memory positions are recomputed backward from this bounded right edge every call (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:68-78`). Stored K is pre-RoPE in relative mode, so the same cache tensors can be re-rotated at their newly compressed relative IDs on every read (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:124-134`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:79-108`). Temporal IDs therefore never grow with video length.

### Train/inference consistency and an explicit limitation

Long-memory training is not a separate no-cache FlexAttention formulation. The specialized class rejects `kv_cache=None` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:23-25`), while streaming training invokes the generator with persistent KV cache for denoising and context commits (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/streaming_switch_training.py:47-67`). Consequently the memory-enabled training rollout and inference use the same cached attention forward and the same relative-position function. Training also periodically detaches recurrent query state according to `bptt_clips` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/streaming_training.py:212-217`); the root recipe sets `bptt_clips=1` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:120-123`).

**Absent:** a memory-aware no-cache training mask/path in `causal_model_infinity_memory.py`. The inherited/base no-cache path does not provide equivalent memory insertion because the specialization replaces self-attention with a function that raises when cache is absent (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:13-25`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:161-168`). Also, memory is deliberately skipped on bulk recache forwards, so recache and normal 3-frame rollout do not have identical tier participation (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:65-78`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:102-108`).

## 7. Timestep conditioning of memory tokens

### Query-memory tokens

Learnable query-memory tokens receive **no timestep embedding or AdaLN modulation at all**. `QueryMemoryEncoder.update()` accepts only evicted K/V and optional sink K/V (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:124-142`); its cross-attention layers use their own RMSNorm/residual/FFN stack without `e` or `t` inputs (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:8-34`). `get_kv()` merely projects the evolved hidden state (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:194-212`). Thus there is no “t=0 embedding” attached directly to Q-memory.

The **write source**, however, comes from KV committed by the context pass. In ordinary inference, after denoising a block, the pipeline calls the generator once more on `denoised_pred` with `context_timestep = context_noise` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:83-100`). Root inference configs set `context_noise: 0` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long_inference.yaml:23-26`). Therefore, in the standard inference recipe, evicted source KV was computed from tokens conditioned at `t=0`; this is indirect source conditioning, not memory-token conditioning.

### Experimental evolving SinkMemory tokens

`SinkMemory.update()` is timestep-conditioned differently: it receives current `e0`, combines each block’s modulation with that `e0`, and applies the same six-way AdaLN modulation to `sink_hidden` through self-attention and FFN (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/sink_memory.py:53-69`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/sink_memory.py:79-91`). The caller passes the current forward’s `e0` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:829-836`). In the normal context-commit call, that is usually the configured context timestep, zero in shipped inference (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:93-95`).

## 8. Delta table: `causal_model.py` → `causal_model_infinity.py` → `causal_model_infinity_memory.py`

| Capability | `causal_model_infinity.py` over the simpler causal baseline | `causal_model.py` over `causal_model_infinity.py` | `causal_model_infinity_memory.py` over `causal_model_infinity.py` |
|---|---|---|---|
| Rolling sink+local cache | Implements finite-cache rolling, preserves sink prefix, and attends sink plus recent local (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity.py:96-169`). | Retains rolling but supports pre-RoPE cache when dynamic/tri/relative RoPE is enabled (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:124-168`). | Reimplements a compact inference-only roll/deferred-update path and assumes pre-RoPE K (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:13-63`). |
| RoPE | Uses `block_relativistic_rope` to renumber the cache window from zero; no memory-aware position tier (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity.py:13-29`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity.py:119-147`). | Adds `causal_rope_apply`, DR-RoPE, tri-tier continuous RoPE, bounded relative RoPE, and invariants (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:14-28`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:175-217`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:250-317`). | Imports the bounded relative-position function from `causal_model.py`, applies sink/memory/local rotations explicitly, and enforces the same bounds (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:8-9`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:64-108`). |
| Query memory | Absent from attention and model state. The file contains no `memory_kv` input or encoder setup; its attention signature ends at `sink_recache_after_switch` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity.py:57-57`). | Adds `memory_kv`, query encoder setup/state/reset, per-group dispatch, last-layer eviction writes, and sink→memory→local concatenation (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:80-95`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:283-300`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:561-567`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:726-739`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:816-828`). | Adds the same core query encoder with a smaller monkey-patched specialization: setup/reset, shared/group KVs, last-layer eviction writes, and sink→memory→local attention (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:155-179`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:212-231`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:261-289`). |
| Evolving sink memory | Absent; sink is a fixed cache prefix throughout rolling (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity.py:149-162`). | Adds optional `SinkMemory`, sink QKV/hidden capture, per-layer evolution from per-layer evicted KV, and precedence over query memory at injection (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:91-95`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:568-580`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:716-739`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:805-836`). | Explicitly sets `sink_memory=None` and never provides `setup_sink_memory`; only fixed sink cache plus query memory is present (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:169-179`). |
| Training/no-cache path | Has FlexAttention no-cache branches in self-attention, although model `_forward_train` immediately raises (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity.py:68-95`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity.py:434-436`). | Also retains no-cache attention logic but model `_forward_train` immediately raises (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:96-123`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:842-844`). | Removes it at the patched-attention level: `kv_cache=None` raises immediately (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:23-25`). |
| Integration selection | Wrapper selects this when `use_infinite_attention=True` but `relative_rope=False` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/wan_wrapper.py:110-117`). | Wrapper selects it for causal non-infinite attention, passing all RoPE flags (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/wan_wrapper.py:117-118`). | Wrapper selects it when both `use_infinite_attention=True` and `relative_rope=True`, then monkey-patches every block via `enable_infmem` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/utils/wan_wrapper.py:110-115`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:157-168`). |

## Additional critical absences and implementation caveats

1. **No use of config `memory_kwargs.rope` or `use_evicted_kv` was found in the requested memory implementation.** The recipes declare them (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:114-118`), but `QueryMemoryEncoder` does not read either among its configuration fields (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:38-68`). Actual memory-K RoPE is controlled by model-level relative/tri/DR RoPE logic, not that `memory_kwargs.rope` key (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:102-108`).
2. **No attention mask separates tiers during specialized cached inference.** The attention helper defaults to non-causal and no mask; the concatenated KV is globally visible to each query in the current block (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/attention.py:54-66`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:109-132`).
3. **No direct timestep embedding is assigned to query memory.** Only evicted source KV reflects the timestep of the context pass (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:124-147`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:93-95`).
4. **Several dimensions are hard-coded to the 1.3B model and 480×832 latent geometry.** Pipeline cache allocation fixes 30 layers, 12 heads, 128 head width, and 1560 tokens/frame (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:28-30`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:141-148`); the specialized eviction path separately hard-codes 1560 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:266-271`).
5. **SinkMemory is not part of the wrapper-selected specialized infinity-memory class.** That class has fixed sink KV plus evolving query memory only (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:169-179`). Experimental evolving sink tokens live in `causal_model.py`/`sink_memory.py` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:568-580`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/sink_memory.py:53-93`).

## INFERRED implications for a Helios port

The following are inferences grounded in the observed implementation, not claims about existing Helios code:

- Because Echo’s query writer consumes already projected last-layer K/V (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:283-288`) and its encoder cross-attention has only a Q projection (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:13-19`), a Helios design that writes hidden states would not be a literal port; it would need K/V projection logic or a redesigned encoder interface.
- Echo’s single query state is reused at every layer through one shared memory K/V projection (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:228-230`), so per-layer Helios memory projections would materially change parameterization and semantics.
- Echo’s bounded IDs depend on preserving pre-RoPE K and re-rotating every active tier each call (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:124-134`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:79-108`); a no-KV-cache latent-history system can preserve the bounded layout, but must assign equivalent frame IDs after each re-patchification.
- Echo’s update becomes visible one forward after eviction because memory K/V is collected before blocks and query state is updated after blocks (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:219-231`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:261-289`). A Helios implementation should make that latency an explicit choice rather than accidentally injecting a same-step write.

## TL;DR

- Shipped long settings are 3 sink frames, 3 query-memory frames, and a 12-frame cache capacity that leaves 9 recent frames, at 1560 tokens/frame (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:55-62`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/configs/echo_infinity-long.yaml:106-114`).
- Query state is `[B,4680,1536]`; memory K/V is `[B,4680,12,128]`; each rolling per-layer cache is `[B,18720,12,128]` under the long recipe (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:40-55`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:194-212`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:60-69`).
- `query_init` is one learned Gaussian-initialized `nn.Parameter`; default one-group mode shares its evolved K/V across all layers (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:67-68`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:82-91`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:228-230`).
- Eviction removes oldest non-sink local frames when the finite cache would overflow (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:34-49`).
- Query-memory writes are sourced from the last transformer layer’s pre-roll, pre-RoPE cached K/V, not hidden states (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:261-289`).
- The writer is two 1536-wide, 12-head cross-attention+FFN layers; cached K/V enter without encoder K/V projections (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:8-34`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:40-68`).
- Update is `g=sigmoid(W[old;candidate])`, `new=g·old+(1−g)·candidate`; gate bias +2 initially retains about 88.1% (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:83-87`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:144-155`).
- Dedicated encoder `to_k/to_v` projections are shared across DiT layers by default; they are not block self-attention projections (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:88-91`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:228-230`).
- Injection order is `[sink, memory, local]`, with no explicit/causal mask in cached attention (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:109-132`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/attention.py:54-66`).
- Bounded relative RoPE clamps the right edge at `pmax−1` and packs memory immediately before local; steady-state IDs are sink 0..2, memory 9..11, local 12..20 (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model.py:63-78`).
- Query memory has no timestep embedding; only its evicted source KV is normally produced by a `context_noise=0` commit (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/query_memory.py:124-147`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/pipeline/causal_inference.py:93-95`).
- The wrapper-selected infinity-memory path is cache-only and lacks evolving `SinkMemory`; that experimental per-layer sink writer exists only in `causal_model.py`/`sink_memory.py` (`/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:23-25`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/wan/modules/causal_model_infinity_memory.py:169-179`, `/mnt/beegfs/siyuan/workspace/Echo-Infinity/model/sink_memory.py:53-93`).

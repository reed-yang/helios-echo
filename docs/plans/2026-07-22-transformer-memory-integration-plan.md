# Transformer Memory Integration Plan (read path + capture API)

> Executed inline by the plan author immediately after writing (deviation from the full-code-in-plan convention: all edits are anchor-precise diffs to one file the author holds in context; the code lives in the commits).

**Spec:** design doc ch.1 (D5-D11, §3-§5, §7), ch.4 R0/D1 arbitration. Predecessor: `2026-07-22-k0-memory-foundations-plan.md`.

**Goal:** Register `HeliosMemoryEncoder` in `HeliosTransformer3DModel`, add the token-insertion read path (memory prefix + fractional RoPE + t=0 AdaLN + independent memory amp + three-way length bookkeeping), and the `capture_last_hidden` API — with existing behavior byte-identical when the feature is off.

## Verified facts this plan rests on

- Target lineage (`stage1_lora_cfr_368_correct.yaml`): `zero_history_timestep: true` (:181), `guidance_cross_attn: true` (:182), `restrict_self_attn: false` (:183), `is_amplify_history: false` (:173) → v1 asserts `zero_history_timestep` when memory is on and reuses the existing t0 span (`transformer_helios.py:1353-1366` computes it from `history_context_length`, which includes the memory prefix once prepended before :1351).
- `attn_varlen_func` has an SDPA fallback chain (`helios_kernels/attention_dispatch.py:132-149`) → CPU unit tests monkeypatch `transformer_helios.attn_varlen_func` to the SDPA form; GPU smoke runs the real kernels.
- **Design D11 correction:** 4 call sites tuple-unpack the 2-tuple return (`utils_helios_post.py:329,3212,3295,3350`) — the design's "constant 3-tuple, all callers index [0]" claim is wrong. v1 returns `(output, logits, last_hidden)` **only when** `capture_last_hidden=True`; all existing call sites untouched.

## Edits (all in `helios/modules/transformer_helios.py` unless noted)

1. Import `HeliosMemoryEncoder`; append `"memory_key_scale"` to `_keep_in_fp32_modules` (:954-961).
2. `HeliosAttention.__init__` (:462): new `is_amplify_memory=False` → `memory_key_scale = Parameter(full((heads,), -4.0))` (near-neutral scale ≈1.16 vs history amp's trained-press 7.58) + `get_scale_memory()` mirroring :530-537 without the frozen cache.
3. `HeliosAttnProcessor.__call__` (:196): new `memory_context_length: int = 0`; `history_seq_len` formula (:213) subtracts it; v1 guard `memory ⇒ not restrict_self_attn and not enable_navit`; amplify-history non-NAViT slice (:408) offset by memory prefix; new independent memory-amp branch scaling `key[:, :memory_context_length]`.
4. `HeliosTransformerBlock` (:718): `is_amplify_memory` ctor passthrough; `forward` gains `memory_context_length=0` → attn1 kwarg; `guidance_cross_attn` branch (:824, :866-869) subtracts memory from `history_seq_len` and splits at `prefix_len = memory + history` (memory stays out of text cross-attn, design D7).
5. `HeliosTransformer3DModel.__init__` (:982): config kwargs `is_enable_evolving_memory / memory_num_query_frames=3 / memory_frame_hw=(12,20) / memory_enc_num_layers=2 / memory_gate_init_bias=0.75 / is_amplify_memory`; construct `self.evolving_memory` AFTER `init_weights()` (custom inits must survive); blocks get `is_amplify_memory`.
6. `_build_memory_rope(batch_size, device)`: fractional ids `(i+1)/(N+1)` on the fixed `memory_frame_hw` grid, mid-tier-style pooled frequencies (`rope` at 2× grid then `center_down_sample_3d(·,(1,2,2))`); slot choice is ablation A9.
7. `forward` (:1271): `memory_tokens=None, capture_last_hidden=False`; entry asserts (history mode required, `zero_history_timestep` required, no NAViT, capture ⇒ `return_dict=False`); prepend memory tokens+rope after `process_input_hidden_states` and before :1351 (t0 span then covers memory for free); thread `memory_context_length` through both block-loop branches; capture `hidden_states[:, -original_context_length:, :].detach()` after the loop; conditional 3-tuple return.

## Tests — `tests/test_transformer_memory.py` (tiny config, CPU via SDPA monkeypatch; identical suite re-run on GPU)

Tiny config: heads 2 × head_dim 8, rope_dim (4,2,2), 2 layers, in/out 4ch, ffn 32, text 32; history = short 1f / mid 2f / long 4f at 8×8 latent; memory N=2 × (2,2) grid → M=8; `is_amplify_memory=True`, `guidance_cross_attn=True`, `zero_history_timestep=True`.

1. Registration: `evolving_memory.*` and `blocks.*.attn1.memory_key_scale` in `named_parameters`; legacy (memory-off) state_dict loads with `strict=False` missing ONLY memory keys.
2. Off-path equivalence: memory-enabled model with `memory_tokens=None` == baseline model (shared weights copied) — bitwise equal output.
3. Read path: passing `get_tokens()` changes the output; output/logits shapes unchanged.
4. Capture API: 3-tuple with `last_hidden [B, L_noisy, dim]` when requested; 2-tuple otherwise.
5. K0 end-to-end gradient: reset → update(dummy evicted) → tokens → model forward → loss.backward() ⇒ nonzero grads on `evolving_memory.{ctx_k_proj, gate_linear, query_init}` AND `memory_key_scale` (read path through the full transformer).
6. `get_scale_memory()` ≈ 1.1636 at init.

## Smoke gates

- CPU: `PYTHONPATH=. $PY -m unittest tests.test_transformer_memory -v` green on login node.
- GPU: same suite green via `srun --gres=gpu:h200:1` on an idle node (NOT c-node07, rwtag-pinned), real flash-attn path.
- Regression: full `tests/` suite green; `train_helios.py` still parses; no existing-call-site edits anywhere.

Deferred to the next plan: trainer wiring (PEFT exclude, trainable_modules, save/load_extra_components section 5, param groups), pipeline section loop, diffusers_version mirror (M2).

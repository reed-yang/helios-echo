# Trainer batch preparation anchors for D5/D7 write-forward reuse

## Conclusion

The Stage-1 flow-loss path has one canonical clean-history constructor: `prepare_stage1_clean_input_from_latents`. For the configured `history_sizes=[16,2,1]`, it sorts sizes descending, splits the 19-frame dataset history temporally as `long=[:16]`, `mid=[16:18]`, and `1x=[18:19]`, then constructs `short=[x0 anchor, 1x]`. Thus the transformer receives 16 long frames, 2 mid frames, and 2 short frames (the first short frame is the separate `x0_latents` anchor; the second is the final 1x history frame). An evicted-history D5 write forward must call this same helper with its 19-frame `evicted_history_latents`, its 9-frame `evicted_latents` as target/model input, and a separately selected anchor. Do not reproduce tier arithmetic independently.

The current Stage-1 `_flow_loss` path neither consumes the dataset's `evicted_*` fields nor reads `memory_single_write_prob`, `memory_bptt_sections`, or `memory_write_source`. Dataset plumbing already emits evicted data when `is_train_memory_module=True`, but the ordinary per-step unpacking discards the complete batch after extracting only the legacy fields. D5/D7 must add the write/unroll consumption explicitly.

All line references are current anchors on branch `echo-memory`.

---

## Card 1 — Canonical Stage-1 history tier split and x0 anchor

**Claim.** The trainer loads `batch["history_latents"]`, `batch["target_latents"]`, and `batch["x0_latents"]`, then delegates splitting to `prepare_stage1_clean_input_from_latents`. That helper sorts the configured sizes descending; for `[16,2,1]` the temporal split is long 16, mid 2, and 1x 1. With `is_keep_x0=True`, it prefixes `x0_latents` to the 1x tensor to form the 2-frame short tier.

**Trainer call anchor — `train_helios.py:1395-1425`.**

```python
elif args.data_config.use_stage1_dataset:
    # Prepare prompt embeds
    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)

    # Prepare stage1 clean data
    history_latents = batch["history_latents"].to(accelerator.device)
    target_latents = batch["target_latents"].to(accelerator.device)
    x0_latents = batch["x0_latents"].to(accelerator.device)
    (
        model_input,
        ...
    ) = prepare_stage1_clean_input_from_latents(
        history_latents=history_latents,
        target_latents=target_latents,
        x0_latents=x0_latents,
        latent_window_size=latent_window_size,
        history_sizes=args.training_config.history_sizes,
        ...
        is_keep_x0=True,
```

**Helper signature/anchor — `helios/utils/utils_helios_base.py:627-640`.**

```python
def prepare_stage1_clean_input_from_latents(
    history_latents,  # VAE latents, (B, C_latent, F_latent, H_latent, W_latent)
    target_latents,
    x0_latents=None,
    latent_window_size: int = 9,
    history_sizes: list = [16, 2, 1],
    ...
    is_keep_x0: bool = True,
```

**Tier construction anchor — `helios/utils/utils_helios_base.py:641-667`.**

```python
if is_keep_x0:
    latents_prefix = x0_latents.to(device, dtype=dtype)
...
history_sizes = sorted(history_sizes, reverse=True)  # From big to small
history_window_size = sum(history_sizes)
...
latents_history_long, latents_history_mid, latents_history_1x = history_latents.split(history_sizes, dim=2)
```

**Short-tier anchor — `helios/utils/utils_helios_base.py:727-740`.**

```python
if is_keep_x0:
    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
else:
    latents_history_short = latents_history_1x

return (
    target_latents,
    ...
    latents_history_short,
    latents_history_mid,
    latents_history_long,
)
```

**Dataset provenance/shape construction — `helios/dataset/dataloader_history_latents_dist.py:219-281`.** The dataset sets `history_window_size = sum(self.history_sizes)`, flattens consecutive sections, prepends 19 zero frames, and slices `history_latent` as the preceding 19 frames and `target_latent` as the following 9 frames. It obtains x0 separately from the first source section.

```python
x0_latent = source_latent[0, :, :1, :, :].clone()
...
history_window_size = sum(self.history_sizes)
section_size = history_window_size + latent_window_size
...
history_latent = continue_source_latent[:, start_indice : start_indice + history_window_size, :, :]
target_latent = continue_vae_latent[:, start_indice + history_window_size : end_indice, :, :]
```

**D5/D7 reuse consequence.** The 19-frame `evicted_history_latents` provided by the dataset has the same preceding-history role and should be passed into this helper rather than directly attempting to create long/mid/short tiers. The helper's target-length assertion requires a 9-frame target/write input.

---

## Card 2 — Transformer index tensors

**Claim.** The clean-input helper derives a single ordered index sequence of 29 positions (`1 anchor + 16 long + 2 mid + 1 1x + 9 target`) and splits it. It makes the short indices by concatenating the anchor index with the final 1x index. For `[16,2,1]`, exact position sets are: `short=[0,19]`, `long=[1..16]`, `mid=[17,18]`, and `hidden/current=[20..28]` (each expanded over batch). These are the tensors `_flow_loss` supplies to the transformer.

**Index construction anchor — `helios/utils/utils_helios_base.py:655-665`.**

```python
indices = (
    torch.arange(0, sum([1, *history_sizes, latent_window_size])).unsqueeze(0).expand(target_latents.shape[0], -1)
)
(
    indices_prefix,
    indices_latents_history_long,
    indices_latents_history_mid,
    indices_latents_history_1x,
    indices_hidden_states,
) = indices.split([1, *history_sizes, latent_window_size], dim=1)
indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=1)
```

**Device transfer anchor — `train_helios.py:1459-1477`.**

```python
indices_hidden_states = indices_hidden_states.to(accelerator.device, non_blocking=True)
indices_latents_history_short = indices_latents_history_short.to(
    accelerator.device, non_blocking=True
)
indices_latents_history_mid = indices_latents_history_mid.to(accelerator.device, non_blocking=True)
indices_latents_history_long = indices_latents_history_long.to(
    accelerator.device, non_blocking=True
)
```

**Consumer/call contract — `helios/utils/utils_helios_base.py:49-60`.**

```python
model_pred = transformer(
    hidden_states=noisy_model_input,
    timestep=timesteps,
    encoder_hidden_states=prompt_embeds,
    indices_hidden_states=indices_hidden_states,
    indices_latents_history_short=indices_latents_history_short,
    indices_latents_history_mid=indices_latents_history_mid,
    indices_latents_history_long=indices_latents_history_long,
    latents_history_short=latents_history_short,
    latents_history_mid=latents_history_mid,
    latents_history_long=latents_history_long,
```

---

## Card 3 — Flow-matching noisy inputs, timesteps, sigmas, targets, and list semantics

**Claim.** For the ordinary Stage-1 path, the clean target from the helper is `model_input`. `prepare_stage1_noise_input` samples `torch.randn_like(model_input)`, samples one density-distributed timestep per batch element, selects the associated sigma/timestep schedule entries, forms `noisy=(1-sigma)*model_input_w_error + sigma*noise_w_error`, and uses the rectified-flow target `noise_w_error-model_input`. Its default `return_list=True` deliberately wraps each result in a singleton list. The Stage-1 list is not created by `is_random_drop`; random drop modifies/zeros histories in the clean-input helper. Stage-2 can generate multiple list entries for pyramid stages/sample ratios; `_flow_loss` has a generic loop to consume either regime.

**Stage-1 noise/timestep sampling anchor — `helios/utils/utils_helios_base.py:744-801`.**

```python
def prepare_stage1_noise_input(
    args,
    model_input,
    noise_scheduler,
    ...
    return_list=True,
):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
...
    u = compute_density_for_timestep_sampling(
        weighting_scheme=args.training_config.weighting_scheme,
        batch_size=bsz,
        logit_mean=args.training_config.logit_mean,
        logit_std=args.training_config.logit_std,
        mode_scale=args.training_config.mode_scale,
    )
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
...
    timesteps = noise_scheduler.temp_timesteps[indices].to(
        device=model_input.device, non_blocking=True
    )
...
    sigmas = noise_scheduler.temp_sigmas[indices].flatten()
```

**Flow construction and singleton wrapping — `helios/utils/utils_helios_base.py:881-899`.**

```python
# Get flow-matching target
noisy_model_input = (1.0 - sigmas) * model_input_w_error + sigmas * noise_w_error
target = noise_w_error - model_input

noisy_model_input_list = [noisy_model_input] if return_list else noisy_model_input
sigmas_list = [sigmas] if return_list else sigmas
timesteps_list = [timesteps] if return_list else timesteps
targets_list = [target] if return_list else target
```

**Stage-1 caller anchor — `train_helios.py:1515-1534`.**

```python
(
    noisy_model_input_list,
    sigmas_list,
    timesteps_list,
    targets_list,
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    use_clean_input,
) = prepare_stage1_noise_input(
    args=args,
    model_input=model_input,
    noise_scheduler=noise_scheduler_copy,
    ...
)
```

**Generic list consumer — `helios/utils/utils_helios_base.py:43-47`.**

```python
assert len(noisy_model_input_list) == len(sigmas_list) == len(timesteps_list) == len(targets_list)

for noisy_model_input, sigmas, timesteps, target in zip(
    noisy_model_input_list, sigmas_list, timesteps_list, targets_list
):
```

**Stage-2 multiplicity evidence — `helios/utils/utils_helios_base.py:983-1045`.** Stage-2 loops `for i_s, cur_sample_ratio in zip(...)` and an inner `for _ in range(cur_sample_ratio)`, appending each stage/sample item to all four lists. This is why the loss contract is list-based beyond the Stage-1 singleton.

**Random-drop distinction — `helios/utils/utils_helios_base.py:669-689`.** `is_random_drop` can zero the prefix/tiers, but does not create loss microbatches/lists.

```python
if is_random_drop:
    if random_drop_t2v_ratio != 0 and torch.rand(1).item() <= random_drop_t2v_ratio:
        if is_keep_x0:
            latents_prefix = torch.zeros_like(...)
        latents_history_1x = torch.zeros_like(...)
        latents_history_mid = torch.zeros_like(...)
        latents_history_long = torch.zeros_like(...)
```

---

## Card 4 — t=0 conditioning and history corrupt/saturation ordering

**Claim.** No `zero_history_timestep` logic appears in either clean-input or Stage-1 noise-input batch preparation. The trainer only passes the configuration value while constructing the transformer. `_flow_loss` supplies schedule-sampled timesteps unchanged; the transformer detects `indices_hidden_states` plus `zero_history_timestep`, creates an explicit zero timestep, and applies its conditioning embedding to the entire inserted history context. Therefore a D5 write forward needs to pass an explicit zero timestep itself; it cannot obtain t=0 from prep.

**Trainer construction config anchor — `train_helios.py:311-321`.**

```python
transformer_additional_kwargs = {
    "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
    "zero_history_timestep": args.training_config.zero_history_timestep,
    "restrict_self_attn": args.training_config.restrict_self_attn,
    ...
}
```

**Transformer t=0 construction anchor — `helios/modules/transformer_helios.py:1461-1476`.**

```python
if indices_hidden_states is not None and self.zero_history_timestep:
    if isinstance(timestep, list):
        timestep_t0 = torch.zeros((1), dtype=timestep[0].dtype, device=timestep[0].device)
    else:
        timestep_t0 = torch.zeros((1), dtype=timestep.dtype, device=timestep.device)
    temb_t0, timestep_proj_t0, _ = self.condition_embedder(
        timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
    )
    temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_context_length, -1)
```

**History corruption placement/config anchor — `helios/utils/utils_helios_base.py:803-846`.** It occurs after optional error-recycling injection and before saturation and model-input corruption.

```python
if args.training_config.use_error_recycling:
    (..., latents_history_long, latents_history_mid, latents_history_short, use_clean_input) = apply_error_injection(...)

if args.training_config.corrupt_history and latents_history_short is not None:
    latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        latent_window_size,
        is_keep_x0=True,
        corrupt_mode=args.training_config.corrupt_mode_history,
        ...
    )
```

**Saturation placement/config anchor — `helios/utils/utils_helios_base.py:848-863`.**

```python
if args.training_config.is_add_saturation and latents_history_short is not None:
    latents_history_short, latents_history_mid, latents_history_long = add_saturation_to_history_latents(
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        latent_window_size,
        is_keep_x0=True,
        saturation_ratio_min=args.training_config.saturation_ratio_min,
        saturation_ratio_max=args.training_config.saturation_ratio_max,
        saturation_clean_prob=args.training_config.saturation_ratio_clean_prob,
    )
```

**Transformation behavior anchor — `helios/utils/utils_helios_base.py:283-318, 475-484, 487-508`.** `corrupt_history_latents` supports `noise`, `downsample`, or random selection, can leave histories unchanged according to `noise_corrupt_clean_prob`, preserves the x0 slot when `is_keep_x0=True`, and reconstructs the three tiers at the original lengths. Saturation likewise preserves the x0 slot from active saturation and applies a per-channel mean-centered multiplicative factor to eligible history spans.

```python
if clean_random < noise_corrupt_clean_prob:
    return latents_history_short, latents_history_mid, latents_history_long
...
latents_4x_recovered, latents_2x_recovered, latents_history_short_recovered = mid_latents_history.split(
    [len_4x, len_2x, ori_len_1x], dim=2
)
...
latent_mean = torch.mean(x1, dim=1, keepdim=True)
x1_saturated = (x1 - latent_mean) * sat_factor + latent_mean
```

---

## Card 5 — Exact current `_flow_loss` call region inside accumulation

**Claim.** Batch tensors originate in the Stage-1 branch immediately above the offload/dropout/device/noise preparation region. The actual `_flow_loss` call is inside `with accelerator.accumulate(models_to_accumulate)`, after a list-length assertion. It receives every clean index/tier tensor plus the prepared noisy lists.

**Batch extraction immediately above — `train_helios.py:1395-1428`.**

```python
elif args.data_config.use_stage1_dataset:
    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)

    history_latents = batch["history_latents"].to(accelerator.device)
    target_latents = batch["target_latents"].to(accelerator.device)
    x0_latents = batch["x0_latents"].to(accelerator.device)
    (
        model_input,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
    ) = prepare_stage1_clean_input_from_latents(...)
```

**Noise preparation and accumulation/call anchor — `train_helios.py:1515-1565`.**

```python
) = prepare_stage1_noise_input(
    args=args,
    model_input=model_input,
    noise_scheduler=noise_scheduler_copy,
    recycle_vars=recycle_vars,
    latents_history_short=latents_history_short,
    latents_history_mid=latents_history_mid,
    latents_history_long=latents_history_long,
    latent_window_size=latent_window_size,
    is_keep_x0=True,
)

with accelerator.accumulate(models_to_accumulate):
    # Predict the noise residual
    if not args.training_config.is_train_dmd and not args.training_config.is_use_ode_regression:
        assert len(noisy_model_input_list) == len(sigmas_list) == len(timesteps_list) == len(targets_list)
        logs = _flow_loss(
            args=args,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            transformer=transformer,
            prompt_embeds=prompt_embeds,
            prompt_attention_masks=None,
            noisy_model_input_list=noisy_model_input_list,
            sigmas_list=sigmas_list,
            timesteps_list=timesteps_list,
            targets_list=targets_list,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            ...
        )
```

**Important existing limitation.** The trainer executes `batch = None; del batch` at `train_helios.py:1435-1436`. Although the Stage-1 dataset can emit `evicted_latents`, `evicted_history_latents`, and `evicted_valid_frames`, the current main-loop unpacking never saves those fields before this deletion.

---

## Card 6 — Prompt embedding preparation for `_flow_loss`

**Claim.** On the Stage-1 flow path, `prompt_embeds` originates directly from `batch["prompt_embeds"]` and is moved to the accelerator device. The only subsequent transformation before `_flow_loss` is classifier-free caption dropout that zeroes complete selected batch rows; it is then moved again (without dtype conversion at this second site). There is no I2V prompt swap or negative-prompt swap in this path. `negative_prompt_embeds` is independently encoded at setup and is used by the DMD generator call, not `_flow_loss`.

**Dataset field evidence — `helios/dataset/dataloader_history_latents_dist.py:480-494`.**

```python
output_dict = {
    ...
    "x0_latents": x0_latent,
    "history_latents": history_latent,
    "target_latents": target_latent,
    ...
    "prompt_embeds": self._pick_prompt_embed(feature_data, choice_idx),
    "prompt_attention_masks": feature_data.get("prompt_attention_mask", None),
}
```

**Trainer direct load/dropout/device anchors — `train_helios.py:1395-1397, 1450-1456, 1478-1479`.**

```python
prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
...
if prompt_embeds is not None:
    dropout_mask = (
        torch.rand(prompt_embeds.shape[0], device=prompt_embeds.device)
        < args.data_config.caption_dropout_p
    )
    prompt_embeds[dropout_mask] = 0
...
if prompt_embeds is not None:
    prompt_embeds = prompt_embeds.to(accelerator.device, non_blocking=True)
```

**Negative-prompt scope evidence — `train_helios.py:302-309, 1708-1714`.**

```python
# For negative prompt
with torch.no_grad():
    negative_prompt_embeds, _ = encode_prompt(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=args.data_config.negative_prompt,
        device=accelerator.device,
    )
```

```python
prompt_embeds=prompt_embeds,
negative_prompt_embeds=negative_prompt_embeds,
# For VRAM manager
```

The latter excerpt is in the DMD path, whereas the flow call above passes only `prompt_embeds` and hardcodes `prompt_attention_masks=None` (`train_helios.py:1540-1546`).

---

## Card 7 — Evolving-memory flag status in the trainer

**Claim.** `memory_single_write_prob` is defined in configuration but has no code consumer in `train_helios.py`; searching the repository finds only its configuration declaration and design/plan text. The hot training-loop region (`train_helios.py:1200-2050`) has no `memory_` reference at all. Existing trainer wiring is setup/config/dataset wiring, not D5/D7 forward/loss wiring.

**Defined-but-unwired flag anchor — `helios/utils/train_config.py:455-471`.**

```python
# ---- Evolving memory (Echo-Infinity port) ----
is_enable_evolving_memory: bool = field(default=False)
...
memory_bptt_sections: int = field(default=1)
memory_write_source: str = field(default="t0_hidden")
memory_single_write_prob: float = field(default=0.0)
memory_tf_unroll: bool = field(default=False)
memory_unroll_sections: int = field(default=1)
```

**Existing dataset setup reads — `train_helios.py:915-931`.**

```python
"return_all_vae_latent": (
    args.training_config.dmd_teacher_forcing and args.training_config.dmd_teacher_forcing_ratio > 0
)
or args.training_config.is_use_gan
or args.training_config.memory_tf_unroll,
"history_sizes": args.training_config.history_sizes,
"is_keep_x0": True,
...
"return_evicted_latent": args.training_config.is_train_memory_module,
}
if args.training_config.memory_tf_unroll:
    dataset_kwargs["num_rollout_sections"] = args.training_config.memory_unroll_sections
    dataset_kwargs["return_rollout_metadata"] = True
```

**Dataset output already available — `helios/dataset/dataloader_history_latents_dist.py:499-517`.**

```python
if self.return_evicted_latent:
    evicted_latent, evicted_history, evicted_valid_frames = eviction
    output_dict["evicted_latents"] = evicted_latent
    output_dict["evicted_history_latents"] = evicted_history
    output_dict["evicted_valid_frames"] = evicted_valid_frames

if self.return_rollout_metadata:
    output_dict["start_section_idx"] = start_section_idx
    output_dict["section_prompt_embeds"] = [
```

**Other existing trainer wiring, outside the hot loop.**

- `is_train_memory_module`, `is_amplify_memory`, and `memory_freeze_backbone` choose trainable parameter scope at `train_helios.py:482-502`.
- `memory_learning_rate` is passed to role-tagged optimizer groups at `train_helios.py:841-846` and restored on forced-LR resume at `train_helios.py:1132-1141`.
- `memory_tf_unroll` and `memory_unroll_sections` are read only at dataset setup (`:919, :929-931`).
- `is_enable_evolving_memory`, architecture shape flags, and `zero_history_timestep` are read at transformer construction (`train_helios.py:311-321`); they are not propagated as per-step `memory_state`/write arguments to `_flow_loss`.

**Not yet wired in trainer loop (confirmed absence).**

- `memory_single_write_prob`: no Python consumer in the repository beyond the `TrainingConfig` field; no write-vs-static batch decision.
- `memory_bptt_sections`: no loop detach cadence.
- `memory_write_source`: no branch selecting a write source.
- `evicted_latents`, `evicted_history_latents`, `evicted_valid_frames`: emitted by dataset but not unpacked in `train_helios.py` before `batch` deletion.
- `memory_tf_unroll` metadata (`clean_all_latents`, `start_section_idx`, `section_prompt_embeds`): emitted/configured but not consumed by the normal `_flow_loss` block.
- No `memory_state`, `capture_last_hidden`, `transformer.evolving_memory.update`, or memory-write forward exists in the normal trainer loop.

**D5/D7 implementation implication.** Reuse the Card 1 helper for every write-forward conditioning history and retain needed `evicted_*`/unroll fields before `batch` is cleared. The write forward must explicitly set t=0 and supply the same index/tier tensors as the regular read forward. The existing flow call currently has no parameter through which to pass evolving-memory state or captured write hidden, so its section-level primitive/call contract must be extended or refactored.

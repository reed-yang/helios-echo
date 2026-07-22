# Stage A dataloader / trainer anchors — current-code evidence

## Conclusion

**The Stage-1 dataset is `helios/dataset/dataloader_history_latents_dist.py::BucketedFeatureDataset`, not `dataloader_dmd.py`.** `train_helios.py` imports it precisely when `args.data_config.use_stage1_dataset` is true; the requested anchors (three-section default, padded continuation source, seeded `choice_idx`, global-random rollout start, cache, and Stage-1 output) all occur in that file.

Current implementation remains pre-Stage-A-unroll at the requested points: Stage-1 trainer kwargs do **not** pass `num_rollout_sections`; `start_section_idx` is sampled through process-global `random.randint`; it is not returned; and cache identity is only `<folder>/dataset_cache.pkl` rather than incorporating rollout U.

---

## Evidence cards

### 1. Stage-1 dataset constructor and default kwargs

**Claim.** The selected Stage-1 dataset constructor has nine keyword parameters after `self`; `num_rollout_sections` defaults to `3`.

**Anchor.** `helios/dataset/dataloader_history_latents_dist.py:12-25`

```python
class BucketedFeatureDataset(Dataset):
    def __init__(
        self,
        feature_folders,
        history_sizes=[16, 2, 1],
        is_keep_x0=True,
        force_rebuild=False,
        return_all_vae_latent=False,
        return_prompt_raw=False,
        num_rollout_sections=3,
        single_res=False,
        single_height=384,
        single_width=640,
        seed=42,
    ):
```

### 2. Padded continuation source construction

**Claim.** The Stage-1 continuation source prepends `history_window_size` zeros (currently `sum([16,2,1]) == 19`) to real, time-flattened source frames.

**Anchor.** `helios/dataset/dataloader_history_latents_dist.py:142-156`

```python
latent_window_size = source_latent.shape[2]
history_window_size = sum(self.history_sizes)
section_size = history_window_size + latent_window_size

temp_source_latent = rearrange(source_latent, "b c t h w -> c (b t) h w")
zero_padding_source = torch.zeros(
    temp_source_latent.shape[0],
    history_window_size,
    temp_source_latent.shape[2],
    temp_source_latent.shape[3],
    device=temp_source_latent.device,
    dtype=temp_source_latent.dtype,
)
continue_source_latent = torch.cat([zero_padding_source, temp_source_latent], dim=1)
```

### 3. Seeded `choice_idx` versus global-random `start_section_idx` (F3)

**Claim.** `choice_idx` is deterministic for `(base_seed, epoch, idx)`, while rollout `start_section_idx` uses process-global Python RNG. This is the F3 divergence called out in the plan.

**Anchor.** `helios/dataset/dataloader_history_latents_dist.py:169-186`

```python
sample_seed = self.base_seed + self._epoch * 1000000 + idx
choice_idx = torch.randint(
    0, total_sections, (1,), generator=torch.Generator().manual_seed(sample_seed)
).item()
if choice_idx == 0 and x0_latent is not None:
    x0_latent = torch.zeros_like(x0_latent)

clean_all_vae_latent = None
if self.return_all_vae_latent:
    max_start_idx = total_sections - self.num_rollout_sections
    if max_start_idx < 0:
        raise ValueError(
            f"Not enough sections: total_sections={total_sections}, num_rollout_sections={self.num_rollout_sections}"
        )
    start_section_idx = random.randint(0, max_start_idx)
    start_indice = start_section_idx * latent_window_size
    end_indice = start_indice + history_window_size + self.num_rollout_sections * latent_window_size
    clean_all_vae_latent = continue_source_latent[:, start_indice:end_indice, :, :]
```

### 4. Choice-section history slice

**Claim.** For `choice_idx = k`, the history is selected from padded continuation coordinates `[9k, 9k+19)` (with `latent_window_size=9`, `history_window_size=19`).

**Anchor.** `helios/dataset/dataloader_history_latents_dist.py:188-192`

```python
start_indice = choice_idx * latent_window_size
end_indice = start_indice + section_size

history_latent = continue_source_latent[:, start_indice : start_indice + history_window_size, :, :]
target_latent = continue_vae_latent[:, start_indice + history_window_size : end_indice, :, :]
```

### 5. Prompt selection binds event prompt to `choice_idx`

**Claim.** `_pick_prompt_embed` accepts the sampled chunk index; for multi-event samples it maps `choice_idx` through `event_idx_per_chunk`, and `__getitem__` passes its sampled `choice_idx` into this method.

**Anchors.** `helios/dataset/dataloader_history_latents_dist.py:282-303`; `:365-379`

```python
def _pick_prompt_embed(self, feature_data, choice_idx=None):
    ...
    if (
        choice_idx is not None
        and "event_idx_per_chunk" in feature_data
        and "event_prompt_embeds" in feature_data
    ):
        event_per_chunk = feature_data["event_idx_per_chunk"]
        event_idx = int(event_per_chunk[min(int(choice_idx), len(event_per_chunk) - 1)].item())
        return feature_data["event_prompt_embeds"][event_idx]
    available = [v for v in self.caption_versions if f"prompt_embed_{v}" in feature_data]
    if available:
        return feature_data[f"prompt_embed_{random.choice(available)}"]
    return feature_data["prompt_embed"]
```

```python
output_dict = {
    ...
    "clean_all_latents": clean_all_vae_latent,
    "prompt_embeds": self._pick_prompt_embed(feature_data, choice_idx),
    "prompt_attention_masks": feature_data.get("prompt_attention_mask", None),
}
```

### 6. Per-folder metadata cache and present cache identity

**Claim.** Each feature folder uses exactly `<folder>/dataset_cache.pkl`. The serialized payload is only `samples` and `buckets`; the cache filename/payload does not include `num_rollout_sections`, history sizes, or any schema/version key today.

**Anchor.** `helios/dataset/dataloader_history_latents_dist.py:51-72`

```python
for folder in self.feature_folders:
    cache_file = os.path.join(folder, "dataset_cache.pkl")
    self._process_folder(folder, cache_file)

def _process_folder(self, folder, cache_file):
    if self.force_rebuild or not os.path.exists(cache_file):
        ...
        cached_data = {"samples": folder_samples, "buckets": folder_buckets}
        if not self.force_rebuild:
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
    else:
        ...
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
        folder_samples = cached_data["samples"]
        folder_buckets = cached_data["buckets"]
```

### 7. Metadata frame filter and resolution buckets

**Claim.** Index construction only rejects `num_frame < 121`; when `single_res` is enabled it permits full, half, and quarter resolution relative to configured dimensions. Bucket keys contain `(num_frame, height, width)`.

**Anchor.** `helios/dataset/dataloader_history_latents_dist.py:98-115`

```python
num_frame = int(parts[-3])
height = int(parts[-2])
width = int(parts[-1].replace(".pt", ""))

# keep length >= 121
if num_frame < 121:
    continue

# keep resolution
allowed_resolutions = [
    (self.single_height, self.single_width),
    (self.single_height // 2, self.single_width // 2),
    (self.single_height // 4, self.single_width // 4),
]
if self.single_res and (height, width) not in allowed_resolutions:
    continue

bucket_key = (num_frame, height, width)
```

### 8. Complete current `__getitem__` output keys

**Claim.** The standard output contains 12 keys; `prompt_raws` is conditionally added when `return_prompt_raw` is enabled. It does not return `start_section_idx` or an unrolled prompt list.

**Anchor.** `helios/dataset/dataloader_history_latents_dist.py:365-384`

```python
output_dict = {
    "uttid": sample_info["uttid"],
    "bucket_key": sample_info["bucket_key"],
    "dataset_name": sample_info["dataset_name"],
    "num_frame": sample_info["num_frame"],
    "height": sample_info["height"],
    "width": sample_info["width"],
    "x0_latents": x0_latent,
    "history_latents": history_latent,
    "target_latents": target_latent,
    "clean_all_latents": clean_all_vae_latent,
    "prompt_embeds": self._pick_prompt_embed(feature_data, choice_idx),
    "prompt_attention_masks": feature_data.get("prompt_attention_mask", None),
}

if self.return_prompt_raw:
    output_dict["prompt_raws"] = prompt_raws

return output_dict
```

### 9. Stage-1 dispatch and current trainer dataset kwargs

**Claim.** `use_stage1_dataset` dispatches to the history-latent dataset. Its trainer kwargs omit `num_rollout_sections`, so its constructor default (`3`) is in effect when the Stage-1 path is selected.

**Anchors.** `train_helios.py:112-129`; `:875-923`

```python
def main(args):
    if args.data_config.use_stage3_dataset:
        from helios.dataset.dataloader_dmd import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )
    elif args.data_config.use_stage1_dataset:
        from helios.dataset.dataloader_history_latents_dist import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )
    else:
        from helios.dataset.dataloader_mp4_dist import (
            BucketedFeatureDataset,
            BucketedSampler,
            collate_fn,
        )
```

```python
# Dataset and DataLoaders creation:
...
elif args.data_config.use_stage1_dataset:
    dataset_kwargs = {
        "feature_folders": args.data_config.instance_data_root,
        "single_res": args.data_config.single_res,
        "single_height": args.data_config.single_height,
        "single_width": args.data_config.single_width,
        "return_prompt_raw": args.training_config.is_use_reward_model,
        "return_all_vae_latent": (
            args.training_config.dmd_teacher_forcing and args.training_config.dmd_teacher_forcing_ratio > 0
        )
        or args.training_config.is_use_gan,
        "history_sizes": args.training_config.history_sizes,
        "is_keep_x0": True,
        "force_rebuild": args.data_config.force_rebuild,
        "seed": args.seed,
    }
...
train_dataset = BucketedFeatureDataset(**dataset_kwargs)
```

### 10. `_flow_loss` signature, DDP zero touch, internal backward, and current return

**Claim.** `_flow_loss` has 22 positional parameters after its name and owns `accelerator.backward(loss)` internally. For full finetuning or `HELIOS_DDP_FIND_UNUSED=0`, it zero-touches every trainable transformer parameter. It returns a logs dict containing final `loss`, first scheduler LR, and conditionally `grad_norm`.

**Anchors.** `helios/utils/utils_helios_base.py:20-42`; `:95-107`; `:142-182`

```python
def _flow_loss(
    args,
    accelerator,
    lr_scheduler,
    transformer,
    prompt_embeds,
    prompt_attention_masks,
    noisy_model_input_list,
    sigmas_list,
    timesteps_list,
    targets_list,
    indices_hidden_states,
    latents_history_short,
    indices_latents_history_short,
    latents_history_mid,
    indices_latents_history_mid,
    latents_history_long,
    indices_latents_history_long,
    recycle_vars,
    global_step,
    noise_scheduler_copy,
    use_clean_input,
):
```

```python
# ... some history/multi-term params go ungraded, which DDP (find_unused=False) rejects.
# The zero touch gives every trainable param a (zero) gradient -> safe to disable find_unused.
if getattr(args.model_config, "is_full_finetune", False) or __import__("os").environ.get("HELIOS_DDP_FIND_UNUSED", "1") == "0":
    touch = sum(p.sum() for p in transformer.parameters() if p.requires_grad)
    loss = loss + 0.0 * touch

accelerator.backward(loss)
```

```python
grad_norm = None
if accelerator.sync_gradients:
    params_to_clip = transformer.parameters()
    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.training_config.max_grad_norm)

logs = {
    "loss": loss.detach().item(),
    "lr": lr_scheduler.get_last_lr()[0],
}
if grad_norm is not None:
    logs["grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
...
return logs
```

### 11. Outer `accelerator.accumulate` ownership scope

**Claim.** Current non-DMD/non-ODE flow training enters `accelerator.accumulate(models_to_accumulate)`, calls the internally-backwarding `_flow_loss` inside that scope, then steps/schedules/zeros the optimizer in the same scope.

**Anchor.** `train_helios.py:1528-1557`

```python
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
            recycle_vars=recycle_vars,
            global_step=global_step,
            noise_scheduler_copy=noise_scheduler_copy,
            use_clean_input=use_clean_input,
        )
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

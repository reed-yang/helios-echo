"""HPSv3 frame-level human preference metric."""

from __future__ import annotations

import gc
import pathlib
from typing import Any, List

from common import EvalRow, extract_frames, mean_or_none, min_or_none, top_mean


def score(rows: List[EvalRow], args, cache_root: pathlib.Path) -> None:
    import torch
    import transformers.image_utils as hf_image_utils
    import transformers.trainer as hf_trainer
    import transformers.trainer_pt_utils as hf_trainer_pt_utils

    if not hasattr(hf_image_utils, "VideoInput"):
        # HPSv3 only needs this legacy symbol for annotations at import time.
        hf_image_utils.VideoInput = Any
    if not hasattr(hf_trainer, "nested_concat"):
        hf_trainer.nested_concat = hf_trainer_pt_utils.nested_concat
    if not hasattr(hf_trainer, "DistributedTensorGatherer"):
        # HPSv3 imports this legacy trainer symbol, but its inference path does
        # not use it. Keep the compatibility shim local to our wrapper.
        hf_trainer.DistributedTensorGatherer = object
    if not hasattr(hf_trainer, "SequentialDistributedSampler"):
        # Same compatibility case as above for current transformers releases.
        hf_trainer.SequentialDistributedSampler = object

    from hpsv3 import HPSv3RewardInferencer
    from hpsv3.model.qwen2vl_trainer import Qwen2VLRewardModelBT

    if not getattr(Qwen2VLRewardModelBT, "_rt_rl_compat_init", False):
        original_init = Qwen2VLRewardModelBT.__init__
        original_load_state_dict = Qwen2VLRewardModelBT.load_state_dict
        original_forward = Qwen2VLRewardModelBT.forward

        def compat_init(self, config, *init_args, **init_kwargs):
            init_kwargs.pop("use_cache", None)
            if not hasattr(config, "hidden_size") and hasattr(config, "text_config"):
                config.hidden_size = config.text_config.hidden_size
            return original_init(self, config, *init_args, **init_kwargs)

        def compat_load_state_dict(self, state_dict, *load_args, **load_kwargs):
            target_keys = set(self.state_dict().keys())
            remapped = {}
            for key, value in state_dict.items():
                new_key = key
                if key.startswith("visual."):
                    new_key = f"model.{key}"
                elif key.startswith("model.") and not key.startswith("model.language_model."):
                    candidate = f"model.language_model.{key[len('model.'):]}"
                    if candidate in target_keys:
                        new_key = candidate
                remapped[new_key] = value
            return original_load_state_dict(self, remapped, *load_args, **load_kwargs)

        def compat_forward(self, *forward_args, **forward_kwargs):
            if forward_args:
                return original_forward(self, *forward_args, **forward_kwargs)

            input_ids = forward_kwargs.get("input_ids")
            attention_mask = forward_kwargs.get("attention_mask")
            inputs_embeds = forward_kwargs.get("inputs_embeds")
            return_dict = forward_kwargs.get("return_dict", True)

            outputs = self.model(
                input_ids=input_ids,
                pixel_values=forward_kwargs.get("pixel_values"),
                pixel_values_videos=forward_kwargs.get("pixel_values_videos"),
                image_grid_thw=forward_kwargs.get("image_grid_thw"),
                video_grid_thw=forward_kwargs.get("video_grid_thw"),
                mm_token_type_ids=forward_kwargs.get("mm_token_type_ids"),
                position_ids=forward_kwargs.get("position_ids"),
                attention_mask=attention_mask,
                past_key_values=forward_kwargs.get("past_key_values"),
                inputs_embeds=inputs_embeds,
                use_cache=forward_kwargs.get("use_cache"),
                output_attentions=forward_kwargs.get("output_attentions"),
                output_hidden_states=forward_kwargs.get("output_hidden_states"),
                return_dict=True,
                cache_position=forward_kwargs.get("cache_position"),
            )
            hidden_states = outputs.last_hidden_state
            logits = self.rm_head(hidden_states.to(torch.float32))

            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            elif input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

            if self.reward_token == "last":
                pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            elif self.reward_token == "mean":
                valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
                pooled_logits = torch.stack(
                    [logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)]
                )
            elif self.reward_token == "special":
                special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for special_token_id in self.special_token_ids:
                    special_token_mask = special_token_mask | (input_ids == special_token_id)
                pooled_logits = logits[special_token_mask, ...].view(batch_size, -1)
            else:
                raise ValueError("Invalid reward_token")

            if return_dict:
                return {"logits": pooled_logits}
            return (pooled_logits,)

        Qwen2VLRewardModelBT.__init__ = compat_init
        Qwen2VLRewardModelBT.load_state_dict = compat_load_state_dict
        Qwen2VLRewardModelBT.forward = compat_forward
        Qwen2VLRewardModelBT._rt_rl_compat_init = True

    checkpoint_path = args.hpsv3_checkpoint
    if checkpoint_path and not pathlib.Path(checkpoint_path).exists():
        raise FileNotFoundError(f"HPSv3 checkpoint not found: {checkpoint_path}")

    inferencer = HPSv3RewardInferencer(
        checkpoint_path=checkpoint_path or None,
        device=args.device,
    )

    for row in rows:
        try:
            frame_paths = extract_frames(row.video_path, cache_root, args.frame_samples)
            prompts = [row.prompt] * len(frame_paths)
            scores = []
            for start in range(0, len(frame_paths), args.hpsv3_batch_size):
                chunk_paths = frame_paths[start : start + args.hpsv3_batch_size]
                chunk_prompts = prompts[start : start + args.hpsv3_batch_size]
                with torch.no_grad():
                    rewards = inferencer.reward(chunk_prompts, image_paths=chunk_paths)
                if getattr(rewards, "ndim", 1) == 2:
                    rewards = rewards[:, 0]
                scores.extend(rewards.detach().float().cpu().tolist())
            row.metrics["hpsv3_frame_mean"] = mean_or_none(scores)
            row.metrics["hpsv3_frame_min"] = min_or_none(scores)
            row.metrics["hpsv3_frame_top30_mean"] = top_mean(scores, 0.30)
            row.errors.pop("hpsv3", None)
        except Exception as exc:
            row.errors["hpsv3"] = repr(exc)

    del inferencer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

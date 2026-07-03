#!/usr/bin/env python3
"""Run one metric for Helios full/front/back long-video segment evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import pathlib
from dataclasses import asdict
from typing import List

import numpy as np

from common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROMPT_FILE,
    DEFAULT_VIDEO_ROOT,
    THIRD_PARTY,
    EvalRow,
    add_third_party_paths,
    discover_videos,
    mean_or_none,
    min_or_none,
    top_mean,
)
from io_utils import read_jsonl, write_all, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", required=True, choices=["dover", "pickscore", "hpsv3", "videoalign"])
    parser.add_argument("--video-root", type=pathlib.Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--prompt-file", type=pathlib.Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--run-glob", type=str, default="*")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--weight-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--frame-samples", type=int, default=8)
    parser.add_argument("--frame-chunk-size", type=int, default=33)
    parser.add_argument("--frames-per-chunk", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=0)

    parser.add_argument("--pickscore-processor", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--pickscore-model", type=str, default="yuvalkirstain/PickScore_v1")

    parser.add_argument("--hpsv3-checkpoint", type=str, default="")
    parser.add_argument("--hpsv3-batch-size", type=int, default=2)

    parser.add_argument("--dover-opt", type=str, default=str(THIRD_PARTY / "DOVER/dover.yml"))
    parser.add_argument("--dover-weights", type=str, default="")

    parser.add_argument("--videoalign-checkpoint", type=str, default=str(THIRD_PARTY / "Astrolabe/reward_ckpts/Videoreward"))
    parser.add_argument("--videoalign-batch-size", type=int, default=1)
    return parser.parse_args()


def load_or_create_rows(args: argparse.Namespace, out_dir: pathlib.Path) -> List[EvalRow]:
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        return read_jsonl(results_path)

    items = discover_videos(args.video_root, args.prompt_file, args.run_glob)
    if args.max_videos > 0:
        items = items[: args.max_videos]
    rows = [EvalRow(**asdict(item)) for item in items]
    write_jsonl(rows, out_dir / "manifest.jsonl")
    return rows


def stable_video_hash(path: str, suffix: str) -> str:
    h = hashlib.sha1()
    h.update(os.path.abspath(path).encode("utf-8"))
    h.update(suffix.encode("utf-8"))
    try:
        st = os.stat(path)
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    except OSError:
        pass
    return h.hexdigest()[:16]


def chunk_aligned_indices(total: int, chunk_size: int, frames_per_chunk: int) -> list[int]:
    indices: list[int] = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        length = end - start
        if length <= 0:
            continue
        count = min(frames_per_chunk, length)
        if count == 1:
            offsets = [length // 2]
        else:
            offsets = [round(i * (length - 1) / (count - 1)) for i in range(count)]
        indices.extend(start + int(offset) for offset in offsets)
    return indices


def extract_chunk_frames(video_path: str, cache_root: pathlib.Path, args: argparse.Namespace) -> list[str]:
    from PIL import Image
    import decord

    reader = decord.VideoReader(video_path)
    total = len(reader)
    if total <= 0:
        raise RuntimeError(f"no frames found: {video_path}")

    if args.frame_chunk_size > 0 and args.frames_per_chunk > 0:
        indices = chunk_aligned_indices(total, args.frame_chunk_size, args.frames_per_chunk)
        suffix = f"chunk{args.frame_chunk_size}_k{args.frames_per_chunk}_n{len(indices)}"
    else:
        count = min(args.frame_samples, total)
        indices = np.linspace(0, total - 1, count, dtype=int).tolist()
        suffix = f"uniform_n{len(indices)}"

    out_dir = cache_root / stable_video_hash(video_path, suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f"{i:03d}.jpg" for i in range(len(indices))]
    if all(path.exists() for path in expected):
        return [str(path) for path in expected]

    paths = []
    for out_idx, frame_idx in enumerate(indices):
        frame_path = out_dir / f"{out_idx:03d}.jpg"
        Image.fromarray(reader[frame_idx].asnumpy()).save(frame_path, quality=95)
        paths.append(str(frame_path))
    return paths


def score_hpsv3(rows: List[EvalRow], args: argparse.Namespace, cache_root: pathlib.Path) -> None:
    import torch
    import transformers.image_utils as hf_image_utils
    import transformers.trainer as hf_trainer
    import transformers.trainer_pt_utils as hf_trainer_pt_utils
    from typing import Any

    if not hasattr(hf_image_utils, "VideoInput"):
        hf_image_utils.VideoInput = Any
    if not hasattr(hf_trainer, "nested_concat"):
        hf_trainer.nested_concat = hf_trainer_pt_utils.nested_concat
    if not hasattr(hf_trainer, "DistributedTensorGatherer"):
        hf_trainer.DistributedTensorGatherer = object
    if not hasattr(hf_trainer, "SequentialDistributedSampler"):
        hf_trainer.SequentialDistributedSampler = object

    from hpsv3 import HPSv3RewardInferencer
    from hpsv3.model.qwen2vl_trainer import Qwen2VLRewardModelBT

    if not getattr(Qwen2VLRewardModelBT, "_rt_rl_compat_init", False):
        # transformers>=5 forwards use_cache into __init__ and may omit a top-level
        # hidden_size on the config; mirror metrics/hpsv3_metric.py's compat_init.
        original_init = Qwen2VLRewardModelBT.__init__

        def compat_init(self, config, *init_args, **init_kwargs):
            init_kwargs.pop("use_cache", None)
            if not hasattr(config, "hidden_size") and hasattr(config, "text_config"):
                config.hidden_size = config.text_config.hidden_size
            return original_init(self, config, *init_args, **init_kwargs)

        Qwen2VLRewardModelBT.__init__ = compat_init

        original_forward = Qwen2VLRewardModelBT.forward

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

        Qwen2VLRewardModelBT.forward = compat_forward
        Qwen2VLRewardModelBT._rt_rl_compat_init = True

    original_module_load_state_dict = torch.nn.Module.load_state_dict

    def compat_module_load_state_dict(module, state_dict, *load_args, **load_kwargs):
        if isinstance(module, Qwen2VLRewardModelBT):
            target_keys = set(module.state_dict().keys())
            remapped = {}
            for key, value in state_dict.items():
                new_key = key
                if key.startswith("visual."):
                    candidate = f"model.{key}"
                    if candidate in target_keys:
                        new_key = candidate
                elif key.startswith("model.") and not key.startswith("model.language_model."):
                    candidate = f"model.language_model.{key[len('model.'):]}"
                    if candidate in target_keys:
                        new_key = candidate
                remapped[new_key] = value
            state_dict = remapped
        return original_module_load_state_dict(module, state_dict, *load_args, **load_kwargs)

    checkpoint_path = args.hpsv3_checkpoint
    if checkpoint_path and not pathlib.Path(checkpoint_path).exists():
        raise FileNotFoundError(f"HPSv3 checkpoint not found: {checkpoint_path}")

    torch.nn.Module.load_state_dict = compat_module_load_state_dict
    try:
        inferencer = HPSv3RewardInferencer(checkpoint_path=checkpoint_path or None, device=args.device)
    finally:
        torch.nn.Module.load_state_dict = original_module_load_state_dict

    for row in rows:
        try:
            frame_paths = extract_chunk_frames(row.video_path, cache_root, args)
            prompts = [row.prompt] * len(frame_paths)
            scores = []
            for start in range(0, len(frame_paths), args.hpsv3_batch_size):
                with torch.no_grad():
                    rewards = inferencer.reward(
                        prompts[start : start + args.hpsv3_batch_size],
                        image_paths=frame_paths[start : start + args.hpsv3_batch_size],
                    )
                if getattr(rewards, "ndim", 1) == 2:
                    rewards = rewards[:, 0]
                scores.extend(rewards.detach().float().cpu().tolist())
            row.metrics["hpsv3_frame_mean"] = mean_or_none(scores)
            row.metrics["hpsv3_frame_min"] = min_or_none(scores)
            row.metrics["hpsv3_frame_top30_mean"] = top_mean(scores, 0.30)
        except Exception as exc:
            row.errors["hpsv3"] = repr(exc)

    del inferencer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_pickscore(rows: List[EvalRow], args: argparse.Namespace, cache_root: pathlib.Path) -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    def feature_tensor(features):
        if torch.is_tensor(features):
            return features
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if isinstance(features, (tuple, list)):
            for item in reversed(features):
                if torch.is_tensor(item) and item.ndim == 2:
                    return item
        raise TypeError(f"cannot resolve PickScore features from {type(features)!r}")

    processor = AutoProcessor.from_pretrained(args.pickscore_processor)
    model = AutoModel.from_pretrained(args.pickscore_model).eval().to(args.device)
    for row in rows:
        try:
            frame_paths = extract_chunk_frames(row.video_path, cache_root, args)
            frames = []
            for path in frame_paths:
                with Image.open(path) as image:
                    frames.append(image.convert("RGB").copy())
            image_inputs = processor(images=frames, padding=True, truncation=True, max_length=77, return_tensors="pt").to(args.device)
            text_inputs = processor(text=[row.prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(args.device)
            with torch.no_grad():
                image_embs = F.normalize(feature_tensor(model.get_image_features(**image_inputs)), dim=-1)
                text_embs = F.normalize(feature_tensor(model.get_text_features(**text_inputs)), dim=-1)
                scores = (model.logit_scale.exp() * (text_embs @ image_embs.T)[0]).detach().float().cpu().tolist()
            row.metrics["pickscore_frame_mean"] = mean_or_none(scores)
            row.metrics["pickscore_frame_min"] = min_or_none(scores)
            row.metrics["pickscore_frame_top30_mean"] = top_mean(scores, 0.30)
            row.errors.pop("pickscore", None)
        except Exception as exc:
            row.errors["pickscore"] = repr(exc)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_videoalign(rows: List[EvalRow], args: argparse.Namespace) -> None:
    import torch
    from astrolabe.reward_models.videoalign.wan_inference import VideoVLMRewardInference
    from common import PROJECT_ROOT

    model_path = pathlib.Path(args.videoalign_checkpoint)
    if not model_path.exists():
        raise FileNotFoundError(f"VideoAlign checkpoint dir not found: {model_path}")

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("HELIOS_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("HELIOS_VIDEOALIGN_DISABLE_FLASH_ATTN2", "1")

    dtype = torch.bfloat16 if args.weight_dtype == "bf16" else torch.float16 if args.weight_dtype == "fp16" else torch.float32
    inferencer = VideoVLMRewardInference(
        load_from_pretrained=str(model_path),
        device=args.device,
        dtype=dtype,
    )

    for start in range(0, len(rows), args.videoalign_batch_size):
        batch = rows[start : start + args.videoalign_batch_size]
        try:
            video_paths = [str(pathlib.Path(row.video_path).resolve()) for row in batch]
            prompts = [row.prompt for row in batch]
            with torch.no_grad():
                rewards = inferencer.reward(video_paths, prompts, use_norm=True)
            for idx, row in enumerate(batch):
                row.metrics["videoalign_vq"] = float(rewards[idx]["VQ"])
                row.metrics["videoalign_mq"] = float(rewards[idx]["MQ"])
                row.metrics["videoalign_ta"] = float(rewards[idx]["TA"])
                row.metrics["videoalign_overall"] = float(rewards[idx]["Overall"])
                row.errors.pop("videoalign", None)
        except Exception as exc:
            for row in batch:
                row.errors["videoalign"] = repr(exc)

    del inferencer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    add_third_party_paths()
    out_dir = args.output_root / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / "frame_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    rows = load_or_create_rows(args, out_dir)
    print(f"[helios-segment-metric:{args.metric}] videos     : {len(rows)}")
    print(f"[helios-segment-metric:{args.metric}] output_dir : {out_dir}")

    if args.metric == "hpsv3":
        score_hpsv3(rows, args, cache_root)
    elif args.metric == "pickscore":
        score_pickscore(rows, args, cache_root)
    elif args.metric == "videoalign":
        score_videoalign(rows, args)
    elif args.metric == "dover":
        from metrics import dover_metric

        dover_metric.score(rows, args, cache_root)
    else:
        raise ValueError(args.metric)

    write_all(rows, out_dir)
    print(f"[helios-segment-metric:{args.metric}] done")


if __name__ == "__main__":
    main()

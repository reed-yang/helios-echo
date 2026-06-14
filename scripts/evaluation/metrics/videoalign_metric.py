"""VideoAlign VQ/MQ/TA metric."""

from __future__ import annotations

import gc
import os
import pathlib
from typing import List

from common import EvalRow, PROJECT_ROOT


def score(rows: List[EvalRow], args, cache_root=None) -> None:
    import torch
    from astrolabe.reward_models.videoalign.wan_inference import VideoVLMRewardInference

    model_path = pathlib.Path(args.videoalign_checkpoint)
    if not model_path.exists():
        raise FileNotFoundError(f"VideoAlign checkpoint dir not found: {model_path}")

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("HELIOS_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("HELIOS_VIDEOALIGN_DISABLE_FLASH_ATTN2", "1")

    dtype = torch.bfloat16 if args.weight_dtype == "bf16" else torch.float16 if args.weight_dtype == "fp16" else torch.float32
    inferencer = VideoVLMRewardInference(load_from_pretrained=str(model_path), device=args.device, dtype=dtype)

    for start in range(0, len(rows), args.videoalign_batch_size):
        batch = rows[start : start + args.videoalign_batch_size]
        try:
            with torch.no_grad():
                rewards = inferencer.reward(
                    [row.video_path for row in batch],
                    [row.prompt for row in batch],
                    use_norm=True,
                )
            for row, reward in zip(batch, rewards):
                row.metrics["videoalign_vq"] = float(reward["VQ"])
                row.metrics["videoalign_mq"] = float(reward["MQ"])
                row.metrics["videoalign_ta"] = float(reward["TA"])
                row.metrics["videoalign_overall"] = float(reward["Overall"])
        except Exception as exc:
            for row in batch:
                row.errors["videoalign"] = repr(exc)

    del inferencer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

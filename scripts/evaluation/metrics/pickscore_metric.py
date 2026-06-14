"""PickScore frame-level prompt preference metric."""

from __future__ import annotations

import gc
import pathlib
from typing import List

from common import EvalRow, extract_frames, mean_or_none, min_or_none, top_mean


def _feature_tensor(features):
    """Return the projected CLIP feature tensor across transformers versions."""
    import torch

    if torch.is_tensor(features):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if isinstance(features, (tuple, list)):
        for item in reversed(features):
            if torch.is_tensor(item) and item.ndim == 2:
                return item
    raise TypeError(f"cannot resolve PickScore features from {type(features)!r}")


def score(rows: List[EvalRow], args, cache_root: pathlib.Path) -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.pickscore_processor)
    model = AutoModel.from_pretrained(args.pickscore_model).eval().to(args.device)

    for row in rows:
        try:
            frame_paths = extract_frames(row.video_path, cache_root, args.frame_samples)
            frames = []
            for path in frame_paths:
                with Image.open(path) as image:
                    frames.append(image.convert("RGB").copy())
            image_inputs = processor(
                images=frames,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(args.device)
            text_inputs = processor(
                text=[row.prompt],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(args.device)
            with torch.no_grad():
                image_embs = _feature_tensor(model.get_image_features(**image_inputs))
                image_embs = F.normalize(image_embs, dim=-1)
                text_embs = _feature_tensor(model.get_text_features(**text_inputs))
                text_embs = F.normalize(text_embs, dim=-1)
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

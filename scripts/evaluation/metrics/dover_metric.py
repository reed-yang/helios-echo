"""DOVER no-reference video quality metric."""

from __future__ import annotations

import gc
import pathlib
from typing import Dict, List

from common import EvalRow, PROJECT_ROOT, THIRD_PARTY


def score(rows: List[EvalRow], args, cache_root=None) -> None:
    import numpy as np
    import torch
    import yaml
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    from dover.models import DOVER

    dover_root = THIRD_PARTY / "DOVER"
    opt_path = pathlib.Path(args.dover_opt)
    if not opt_path.is_absolute():
        opt_path = PROJECT_ROOT / opt_path
    if not opt_path.exists():
        raise FileNotFoundError(f"DOVER opt file not found: {opt_path}")

    opt = yaml.safe_load(opt_path.read_text(encoding="utf-8"))
    load_path = pathlib.Path(args.dover_weights) if args.dover_weights else pathlib.Path(opt["test_load_path"])
    if not load_path.is_absolute():
        load_path = dover_root / load_path
    if not load_path.exists():
        raise FileNotFoundError(
            f"DOVER weights not found: {load_path}. Download DOVER.pth/DOVER-Mobile.pth outside third-party "
            "and pass DOVER_WEIGHTS=/path/to/weights.pth if you do not want to modify the DOVER clone."
        )

    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])

    def fuse(results: List[float]) -> Dict[str, float]:
        aesthetic_raw, technical_raw = results[0], results[1]
        t = (technical_raw - 0.1107) / 0.07355
        a = (aesthetic_raw + 0.08285) / 0.03774
        overall = t * 0.6104 + a * 0.3896
        return {
            "dover_aesthetic": float(1 / (1 + np.exp(-a))),
            "dover_technical": float(1 / (1 + np.exp(-t))),
            "dover_overall": float(1 / (1 + np.exp(-overall))),
            "dover_aesthetic_raw": float(aesthetic_raw),
            "dover_technical_raw": float(technical_raw),
        }

    evaluator = DOVER(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(torch.load(load_path, map_location=args.device))
    evaluator.eval()
    dopt = opt["data"]["val-l1080p"]["args"]

    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            temporal_samplers[stype] = UnifiedFrameSampler(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
        else:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )

    for row in rows:
        try:
            views, _ = spatial_temporal_view_decomposition(row.video_path, dopt["sample_types"], temporal_samplers)
            for key, value in views.items():
                num_clips = dopt["sample_types"][key].get("num_clips", 1)
                views[key] = (
                    ((value.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(value.shape[0], num_clips, -1, *value.shape[2:])
                    .transpose(0, 1)
                    .to(args.device)
                )
            with torch.no_grad():
                raw_results = [result.mean().item() for result in evaluator(views)]
            row.metrics.update(fuse(raw_results))
        except Exception as exc:
            row.errors["dover"] = repr(exc)

    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

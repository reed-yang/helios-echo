"""Shared helpers for the long-video segment-drift evaluation backend.

Reconstructed from the exact usage in ``run_helios_long_segment_metric.py``,
``merge_sharded_eval_results.py`` and ``metrics/*.py`` (the upstream
``yushen/long-video-eval`` commit referenced these modules but never committed
them). Keep the public surface stable: those callers import the names below.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# scripts/evaluation/common.py -> repo root is two parents up.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
THIRD_PARTY = PROJECT_ROOT / "third-party"

DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "tmp" / "outputs" / "long_video_eval" / "source_root"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "outputs" / "evaluation-results"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "example" / "prompt.txt"


@dataclass
class EvalRow:
    """One (video, prompt) evaluation item.

    ``run_name`` is the run-glob-relative directory (e.g. ``"step-3000/videos"``);
    its first path segment is the model name. ``video_id`` is the file stem and
    is stable across full/start15/end15 segments so drift can be paired per video.
    """

    video_path: str
    prompt: str = ""
    run_name: str = ""
    video_id: str = ""
    idx: Optional[int] = None
    metrics: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)


def add_third_party_paths() -> None:
    """Make the cloned metric repos importable (DOVER/HPSv3/PickScore)."""
    for sub in ("DOVER", "HPSv3", "PickScore"):
        path = THIRD_PARTY / sub
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_prompts(prompt_file) -> List[str]:
    text = pathlib.Path(prompt_file).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def discover_videos(video_root, prompt_file, run_glob: str = "*") -> List[EvalRow]:
    """Find ``<run_glob>/*.mp4`` videos and line-align each to a prompt by stem.

    ``run_glob`` matches run directories under ``video_root`` (e.g. ``'*/videos'``
    or ``'step-3000/videos'``). Videos must have integer stems (``0000.mp4``);
    prompt line ``idx`` (0-based) is bound to video ``idx``.
    """
    video_root = pathlib.Path(video_root)
    prompts = _load_prompts(prompt_file)
    items: List[EvalRow] = []
    run_dirs = sorted(p for p in video_root.glob(run_glob) if p.is_dir())
    for run_dir in run_dirs:
        run_name = str(run_dir.relative_to(video_root))
        for video in sorted(run_dir.glob("*.mp4")):
            stem = video.stem
            try:
                idx: Optional[int] = int(stem)
            except ValueError:
                idx = None
            prompt = prompts[idx] if (idx is not None and 0 <= idx < len(prompts)) else ""
            items.append(
                EvalRow(
                    video_path=str(video.resolve()),
                    prompt=prompt,
                    run_name=run_name,
                    video_id=stem,
                    idx=idx,
                )
            )
    return items


# ---- numeric aggregations -------------------------------------------------

def _clean(values) -> List[float]:
    return [float(v) for v in values if v is not None]


def mean_or_none(values):
    vals = _clean(values)
    return sum(vals) / len(vals) if vals else None


def min_or_none(values):
    vals = _clean(values)
    return min(vals) if vals else None


def top_mean(values, frac: float):
    """Mean of the top ``frac`` fraction of values (by magnitude, descending)."""
    vals = sorted(_clean(values), reverse=True)
    if not vals:
        return None
    k = max(1, int(round(len(vals) * frac)))
    return sum(vals[:k]) / k


# ---- frame extraction (used by metrics/*.py adapters) ---------------------

def extract_frames(video_path: str, cache_root, num_samples: int) -> List[str]:
    """Uniformly sample ``num_samples`` frames, cache as jpg, return paths.

    The in-tree runner uses its own ``extract_chunk_frames``; this exists for the
    standalone ``metrics/pickscore_metric.py`` / ``hpsv3_metric.py`` adapters.
    """
    import hashlib
    import os

    import numpy as np
    from PIL import Image
    import decord

    reader = decord.VideoReader(str(video_path))
    total = len(reader)
    if total <= 0:
        raise RuntimeError(f"no frames found: {video_path}")
    count = min(num_samples, total)
    indices = np.linspace(0, total - 1, count, dtype=int).tolist()

    h = hashlib.sha1(os.path.abspath(str(video_path)).encode("utf-8"))
    h.update(f"uniform_n{count}".encode("utf-8"))
    out_dir = pathlib.Path(cache_root) / h.hexdigest()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for out_idx, frame_idx in enumerate(indices):
        frame_path = out_dir / f"{out_idx:03d}.jpg"
        if not frame_path.exists():
            Image.fromarray(reader[frame_idx].asnumpy()).save(frame_path, quality=95)
        paths.append(str(frame_path))
    return paths

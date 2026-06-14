"""Input manifest generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import RuntimeConfig


@dataclass
class VideoItem:
    index: int
    model: str
    video_path: str
    prompt: str


def read_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
    return prompts


def write_prompt_jsonl(prompts: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            f.write(json.dumps({"index": idx, "prompt": prompt}, ensure_ascii=False) + "\n")


def video_dir_for_model(runtime: RuntimeConfig, model: str) -> Path:
    return runtime.video_roots.get(model) or runtime.source_root / model / "videos"


def build_manifest(runtime: RuntimeConfig, *, expected_videos: int = 0, strict: bool = False) -> dict:
    if not runtime.video_roots and not runtime.source_root.exists():
        raise FileNotFoundError(f"source_root not found: {runtime.source_root}")
    if not runtime.prompt_file.exists():
        raise FileNotFoundError(f"prompt_file not found: {runtime.prompt_file}")
    if runtime.duration_frames <= 0:
        raise ValueError("duration_frames must be positive")

    prompts = read_prompts(runtime.prompt_file)
    if not prompts:
        raise ValueError(f"prompt file has no non-empty prompts: {runtime.prompt_file}")

    items: list[VideoItem] = []
    model_reports: list[dict] = []
    missing: list[str] = []
    for model in runtime.models:
        video_dir = video_dir_for_model(runtime, model)
        if not video_dir.exists():
            raise FileNotFoundError(f"video dir not found for model={model}: {video_dir}")
        videos = sorted(video_dir.glob("*.mp4"))
        if expected_videos > 0 and len(videos) != expected_videos:
            raise ValueError(f"model={model} expected {expected_videos} videos, found {len(videos)}: {video_dir}")
        for idx, prompt in enumerate(prompts):
            path = video_dir / f"{idx:04d}.mp4"
            if not path.exists():
                missing.append(str(path))
                continue
            items.append(VideoItem(index=idx, model=model, video_path=str(path), prompt=prompt))
        model_reports.append(
            {
                "model": model,
                "video_dir": str(video_dir),
                "video_count": len(videos),
                "matched_prompt_count": len([item for item in items if item.model == model]),
            }
        )

    if missing and strict:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} indexed videos, first entries:\n{preview}")

    manifest = {
        "dataset_name": runtime.dataset_name,
        "source_root": str(runtime.source_root),
        "prompt_file": str(runtime.prompt_file),
        "output_root": str(runtime.output_root),
        "video_roots": {model: str(path) for model, path in runtime.video_roots.items()},
        "duration_frames": runtime.duration_frames,
        "models": runtime.models,
        "gpus": runtime.gpus,
        "num_shards": runtime.num_shards,
        "prompt_count": len(prompts),
        "item_count": len(items),
        "missing_count": len(missing),
        "missing_preview": missing[:20],
        "model_reports": model_reports,
        "items": [asdict(item) for item in items],
    }
    return manifest


def write_manifest(manifest: dict, output_root: Path, dataset_name: str) -> Path:
    path = output_root / "_manifests" / f"{dataset_name}.manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

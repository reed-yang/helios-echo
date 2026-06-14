"""Configuration loading and template expansion."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def deep_format(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(FormatDict({k: str(v) for k, v in context.items()}))
    if isinstance(value, list):
        return [deep_format(item, context) for item in value]
    if isinstance(value, dict):
        return {key: deep_format(item, context) for key, item in value.items()}
    return value


def split_list(value: str | list[str] | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = value.replace(",", " ").split()
    if not items:
        raise ValueError(f"empty list: {value!r}")
    return items


@dataclass
class RuntimeConfig:
    raw: dict[str, Any]
    config_path: Path
    source_root: Path
    prompt_file: Path
    output_root: Path
    dataset_name: str
    duration_frames: int
    models: list[str]
    gpus: list[str]
    cache_dir: Path
    project_root: Path
    video_roots: dict[str, Path]

    @property
    def num_shards(self) -> int:
        return len(self.gpus)

    @property
    def models_space(self) -> str:
        return " ".join(self.models)

    @property
    def gpus_space(self) -> str:
        return " ".join(self.gpus)

    @property
    def gpus_comma(self) -> str:
        return ",".join(self.gpus)


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_dataset_name(source_root: Path) -> str:
    return f"{source_root.parent.name}_{source_root.name}"


def build_runtime(
    config_path: Path,
    *,
    source_root: str | None = None,
    prompt_file: str | None = None,
    output_root: str | None = None,
    dataset_name: str | None = None,
    duration_frames: int | None = None,
    models: str | list[str] | None = None,
    gpus: str | list[str] | None = None,
    cache_dir: str | None = None,
) -> RuntimeConfig:
    raw = load_json_config(config_path)

    base_context = {
        "cwd": str(Path.cwd()),
        "home": str(Path.home()),
        "config_dir": str(config_path.resolve().parent),
    }
    raw = deep_format(raw, base_context)

    defaults = raw.get("defaults", {})
    project_root = Path(defaults.get("project_root") or os.environ.get("PROJECT_ROOT") or Path.cwd()).expanduser().resolve()
    defaults = deep_format(defaults, {**base_context, "project_root": project_root})
    prompt = Path(prompt_file or defaults.get("prompt_file", "")).expanduser()
    output = Path(output_root or defaults.get("output_root", "outputs/long_video_eval")).expanduser()
    cache = Path(cache_dir or defaults.get("cache_dir", "~/.cache/long_video_eval")).expanduser()
    dframes = int(duration_frames or defaults.get("duration_frames", 0))
    video_roots_raw = defaults.get("video_roots") or {}
    if not isinstance(video_roots_raw, dict):
        raise ValueError("defaults.video_roots must be a mapping from model name to video directory")
    video_roots = {
        str(model): Path(str(path)).expanduser()
        for model, path in video_roots_raw.items()
        if str(model).strip() and str(path).strip()
    }
    default_models = list(video_roots) if video_roots else ["base", "distilled"]
    model_list = split_list(models if models is not None else defaults.get("models"), default_models)
    gpu_list = split_list(gpus if gpus is not None else defaults.get("gpus"), ["0"])
    source_default = defaults.get("source_root")
    source_value = source_root or source_default
    if source_value:
        source = Path(source_value).expanduser()
        dname = dataset_name or defaults.get("dataset_name") or infer_dataset_name(source)
    else:
        dname = dataset_name or defaults.get("dataset_name") or "long_video_eval"
        source = output / "_inputs" / dname / "source_root"

    runtime = RuntimeConfig(
        raw=raw,
        config_path=config_path,
        source_root=source,
        prompt_file=prompt,
        output_root=output,
        dataset_name=dname,
        duration_frames=dframes,
        models=model_list,
        gpus=gpu_list,
        cache_dir=cache,
        project_root=project_root,
        video_roots=video_roots,
    )

    context = runtime_context(runtime)
    runtime.raw = deep_format(raw, context)
    return runtime


def runtime_context(runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "project_root": runtime.project_root,
        "source_root": runtime.source_root,
        "prompt_file": runtime.prompt_file,
        "output_root": runtime.output_root,
        "dataset_name": runtime.dataset_name,
        "duration_frames": runtime.duration_frames,
        "models_space": runtime.models_space,
        "gpus_space": runtime.gpus_space,
        "gpus_comma": runtime.gpus_comma,
        "num_shards": runtime.num_shards,
        "cache_dir": runtime.cache_dir,
        "has_video_roots": bool(runtime.video_roots),
        "config_dir": runtime.config_path.resolve().parent,
        "cwd": Path.cwd(),
        "home": Path.home(),
    }

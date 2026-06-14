"""Metric command orchestration."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .config import RuntimeConfig, deep_format, runtime_context
from .manifest import build_manifest, read_prompts, write_manifest, write_prompt_jsonl


def stage_video_roots(runtime: RuntimeConfig, *, dry_run: bool = False) -> Path:
    if not runtime.video_roots:
        return runtime.source_root

    print(f"[runner] backend_source_root={runtime.source_root}")
    for model in runtime.models:
        src_dir = runtime.video_roots.get(model)
        if src_dir is None:
            continue
        dst_dir = runtime.source_root / model / "videos"
        print(f"[runner] input_view {model}: {src_dir} -> {dst_dir}")
        if dry_run:
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_dir.glob("*.mp4")):
            dst = dst_dir / src.name
            if dst.exists() or dst.is_symlink():
                continue
            dst.symlink_to(src.resolve())
    return runtime.source_root


def command_context(runtime: RuntimeConfig, manifest_path: Path, prompt_jsonl: Path) -> dict[str, Any]:
    context = runtime_context(runtime)
    context.update(
        {
            "manifest_json": manifest_path,
            "prompt_jsonl": prompt_jsonl,
            "output_root": runtime.output_root,
            "dataset_name": runtime.dataset_name,
            "models_space": runtime.models_space,
            "gpus_space": runtime.gpus_space,
            "gpus_comma": runtime.gpus_comma,
            "num_shards": runtime.num_shards,
        }
    )
    return context


def run_external_command(step: dict[str, Any], context: dict[str, Any], *, dry_run: bool = False) -> None:
    rendered = deep_format(step, context)
    name = rendered.get("name", "external_command")
    cwd = Path(rendered.get("cwd") or context.get("project_root") or Path.cwd()).expanduser()
    env = os.environ.copy()
    for key, value in rendered.get("env", {}).items():
        env[str(key)] = str(value)

    cmd = rendered.get("command")
    if isinstance(cmd, str):
        printable = cmd
        shell = True
        cmd_value: str | list[str] = cmd
    elif isinstance(cmd, list):
        printable = " ".join(shlex.quote(str(part)) for part in cmd)
        shell = False
        cmd_value = [str(part) for part in cmd]
    else:
        raise ValueError(f"step={name} missing command")

    print(f"[runner] step={name}")
    print(f"[runner] cwd={cwd}")
    print(f"[runner] cmd={printable}")
    if dry_run:
        for key, value in sorted(rendered.get("env", {}).items()):
            print(f"[runner] env {key}={value}")
        return
    subprocess.run(cmd_value, cwd=str(cwd), env=env, shell=shell, check=True)


def run_suite(runtime: RuntimeConfig, *, dry_run: bool = False, strict: bool = False, expected_videos: int = 0) -> Path:
    runtime.output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(runtime, expected_videos=expected_videos, strict=strict)
    manifest_path = write_manifest(manifest, runtime.output_root, runtime.dataset_name)
    prompt_jsonl = runtime.output_root / "_generated_prompts" / f"{runtime.dataset_name}.jsonl"
    write_prompt_jsonl(read_prompts(runtime.prompt_file), prompt_jsonl)
    stage_video_roots(runtime, dry_run=dry_run)
    print(f"[runner] manifest={manifest_path}")
    print(f"[runner] prompt_jsonl={prompt_jsonl}")
    print(json.dumps({k: manifest[k] for k in ["dataset_name", "prompt_count", "item_count", "missing_count"]}, indent=2))

    context = command_context(runtime, manifest_path, prompt_jsonl)
    for step in runtime.raw.get("run", {}).get("steps", []):
        kind = step.get("kind", "external_command")
        enabled = bool(step.get("enabled", True))
        if not enabled:
            print(f"[runner] skip disabled step={step.get('name', kind)}")
            continue
        if kind == "external_command":
            run_external_command(step, context, dry_run=dry_run)
        else:
            raise ValueError(f"unknown run step kind: {kind}")
    return manifest_path

#!/usr/bin/env python3
"""Prepare and merge Helios official eval runs on local prompt/video sets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


METRICS = {
    "aesthetic": {
        "file": "aesthetic_results.json",
        "aggregate_key": "average_score",
        "per_video_key": "aesthetic_score",
        "lower_is_better": False,
    },
    "motion_amplitude": {
        "file": "motion_amplitude_results.json",
        "aggregate_key": "average_score",
        "per_video_key": "motion_fb",
        "lower_is_better": False,
    },
    "motion_smoothness": {
        "file": "motion_smoothness_results.json",
        "aggregate_key": "average_score",
        "per_video_key": "motion_smoothness_score",
        "lower_is_better": False,
    },
    "semantic": {
        "file": "semantic_results.json",
        "aggregate_key": "average_score",
        "per_video_key": "semantic_score",
        "lower_is_better": False,
    },
    "drifting_aesthetic": {
        "file": "drifting_aesthetic_results.json",
        "aggregate_key": "average_drift_score",
        "per_video_key": "drift_aesthetic_score",
        "lower_is_better": True,
    },
    "drifting_motion_smoothness": {
        "file": "drifting_motion_smoothness_results.json",
        "aggregate_key": "average_drift_score",
        "per_video_key": "drift_motion_smoothness_score",
        "lower_is_better": True,
    },
    "drifting_semantic": {
        "file": "drifting_semantic_results.json",
        "aggregate_key": "average_drift_score",
        "per_video_key": "drift_semantic_score",
        "lower_is_better": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--prompt-jsonl", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--dataset-name", required=True)
    prepare.add_argument("--models", nargs="+", default=["base", "mid", "distilled"])
    prepare.add_argument("--num-shards", type=int, default=8)
    prepare.add_argument("--duration-frames", type=int, default=231)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--dataset-name", required=True)
    merge.add_argument("--models", nargs="+", default=["base", "mid", "distilled"])
    merge.add_argument("--num-shards", type=int, default=8)
    merge.add_argument("--metrics", nargs="+", default=list(METRICS))
    merge.add_argument("--helios-merge-script", type=Path, required=True)

    run_metric = subparsers.add_parser("run-metric")
    run_metric.add_argument("--helios-eval-dir", type=Path, required=True)
    run_metric.add_argument("--metric", choices=list(METRICS), required=True)
    run_metric.add_argument("--input-csv", type=Path, required=True)
    run_metric.add_argument("--video-dir", type=Path, required=True)
    run_metric.add_argument("--output-path", type=Path, required=True)
    run_metric.add_argument("--height", type=int, default=384)
    run_metric.add_argument("--width", type=int, default=640)
    run_metric.add_argument("--clip-model-path", type=Path, default=None)
    run_metric.add_argument("--aesthetic-model-path", type=Path, default=None)
    run_metric.add_argument("--smoothness-model-path", type=Path, default=None)
    run_metric.add_argument("--amt-config", type=Path, default=None)
    run_metric.add_argument("--semantic-model-path", type=Path, default=None)
    run_metric.add_argument("--sample-mode", default="middle", choices=["middle", "rand"])
    run_metric.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_index(row: dict[str, Any], fallback: int) -> int:
    value = row.get("index", fallback)
    return int(value)


def symlink_or_replace(target: Path, link_path: Path) -> None:
    target = target.resolve()
    if link_path.is_symlink():
        existing = Path(os.readlink(link_path))
        if not existing.is_absolute():
            existing = (link_path.parent / existing).resolve()
        if existing == target:
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"{link_path} exists and is not a symlink")
    link_path.symlink_to(target)


def prepare(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.prompt_jsonl)
    args.output_root.mkdir(parents=True, exist_ok=True)
    playground_root = args.output_root / "playground"
    playground_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "prompts.csv"
    tasks_path = args.output_root / "tasks.tsv"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "duration"])
        writer.writeheader()
        for fallback, row in enumerate(rows):
            idx = prompt_index(row, fallback)
            writer.writerow(
                {
                    "id": idx + 1,
                    "prompt": row["prompt"],
                    "duration": args.duration_frames,
                }
            )

    tasks: list[dict[str, str]] = []
    missing: list[str] = []
    for model in args.models:
        for shard in range(args.num_shards):
            task_name = f"{args.dataset_name}_{model}_shard_{shard:02d}"
            video_dir = playground_root / task_name
            video_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for fallback, row in enumerate(rows):
                idx = prompt_index(row, fallback)
                if idx % args.num_shards != shard:
                    continue
                src = args.source_root / model / "videos" / f"{idx:04d}.mp4"
                if not src.exists():
                    missing.append(str(src))
                    continue
                video_id = idx + 1
                dst = video_dir / f"{video_id}_{args.duration_frames}_{args.duration_frames}_ori.mp4"
                symlink_or_replace(src, dst)
                count += 1
            tasks.append(
                {
                    "task_name": task_name,
                    "model": model,
                    "shard": f"{shard:02d}",
                    "video_dir": str(video_dir.resolve()),
                    "prompt_csv": str(csv_path.resolve()),
                    "num_videos": str(count),
                }
            )

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} source videos, first entries:\n{preview}")

    with tasks_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_name", "model", "shard", "video_dir", "prompt_csv", "num_videos"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(tasks)

    print(f"[heliosbench-helper] prompts : {csv_path}")
    print(f"[heliosbench-helper] tasks   : {tasks_path}")
    print(f"[heliosbench-helper] videos  : {sum(int(t['num_videos']) for t in tasks)} symlinks")


def load_helios_merge_module(path: Path):
    module_name = "helios_merge_scores"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_script_module(path: Path):
    module_name = "heliosbench_" + path.stem.replace("-", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_metric(args: argparse.Namespace) -> None:
    eval_dir = args.helios_eval_dir.resolve()
    sys.path.insert(0, str(eval_dir))

    script_map = {
        "aesthetic": "0_get_aesthetic.py",
        "motion_amplitude": "1_get_motion_amplitude.py",
        "motion_smoothness": "2_get_motion_smoothness.py",
        "semantic": "3_get_semantic.py",
        "drifting_aesthetic": "5_get_drifting_aesthetic.py",
        "drifting_motion_smoothness": "6_get_drifting_motion_smoothness.py",
        "drifting_semantic": "7_get_drifting_semantic.py",
    }
    script_path = eval_dir / script_map[args.metric]

    if args.metric == "motion_amplitude":
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--input_csv",
                str(args.input_csv.resolve()),
                "--video_dir",
                str(args.video_dir.resolve()),
                "--output_path",
                str(args.output_path.resolve()),
                "--num_workers",
                str(args.num_workers),
            ],
            cwd=eval_dir,
            check=True,
        )
        return

    module = load_script_module(script_path)

    base_kwargs = {
        "height": args.height,
        "width": args.width,
        "input_csv": str(args.input_csv.resolve()),
        "video_dir": str(args.video_dir.resolve()),
        "output_path": str(args.output_path.resolve()),
    }
    metric_kwargs: dict[str, Any] = {}
    if args.metric in {"aesthetic", "drifting_aesthetic"}:
        if args.clip_model_path is None or args.aesthetic_model_path is None:
            raise ValueError(f"{args.metric} requires clip/aesthetic model paths")
        metric_kwargs.update(
            {
                "clip_model_path": str(args.clip_model_path.resolve()),
                "aesthetic_model_path": str(args.aesthetic_model_path.resolve()),
            }
        )
    elif args.metric == "motion_amplitude":
        metric_kwargs["num_workers"] = args.num_workers
    elif args.metric in {"motion_smoothness", "drifting_motion_smoothness"}:
        if args.amt_config is None or args.smoothness_model_path is None:
            raise ValueError(f"{args.metric} requires AMT config/checkpoint paths")
        metric_kwargs.update(
            {
                "config": str(args.amt_config.resolve()),
                "smoothness_model_path": str(args.smoothness_model_path.resolve()),
            }
        )
    elif args.metric in {"semantic", "drifting_semantic"}:
        if args.semantic_model_path is None:
            raise ValueError(f"{args.metric} requires semantic model path")
        metric_kwargs["semantic_model_path"] = str(args.semantic_model_path.resolve())
        if args.metric == "semantic":
            metric_kwargs["sample_mode"] = args.sample_mode

    old_cwd = Path.cwd()
    try:
        os.chdir(eval_dir)
        module.main(argparse.Namespace(**base_kwargs, **metric_kwargs))
    finally:
        os.chdir(old_cwd)


def average(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if key in item and item[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def combine_metric(
    output_root: Path,
    dataset_name: str,
    model: str,
    num_shards: int,
    metric: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    spec = METRICS[metric]
    results_root = output_root / "results"
    per_video: list[dict[str, Any]] = []
    missing: list[str] = []

    for shard in range(num_shards):
        task_name = f"{dataset_name}_{model}_shard_{shard:02d}"
        path = results_root / task_name / spec["file"]
        if not path.exists():
            missing.append(str(path))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        per_video.extend(data.get("per_video_results", []))

    if not per_video:
        return None, missing

    per_video = sorted(per_video, key=lambda item: int(item["id"]))
    score = average(per_video, spec["per_video_key"])
    if score is None:
        return None, missing

    aggregate_key = spec["aggregate_key"]
    combined = {
        "metric": metric,
        aggregate_key: score,
        "num_videos": len(per_video),
        "per_video_results": per_video,
        "source_shards": num_shards,
        "missing_shards": missing,
    }
    if spec["lower_is_better"]:
        combined["description"] = "Lower is better for this drifting metric."
    return combined, missing


def merge(args: argparse.Namespace) -> None:
    merge_module = load_helios_merge_module(args.helios_merge_script)
    merged_root = args.output_root / "results_merged_no_naturalness"
    merged_root.mkdir(parents=True, exist_ok=True)
    all_models: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for model in args.models:
        model_dir = merged_root / model
        model_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {}
        available_weight = 0.0
        weighted_rating = 0.0

        for metric in args.metrics:
            combined, missing = combine_metric(args.output_root, args.dataset_name, model, args.num_shards, metric)
            if combined is None:
                summary[metric] = {"missing": True, "missing_shards": missing}
                continue

            out_path = model_dir / METRICS[metric]["file"]
            out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
            score = combined[METRICS[metric]["aggregate_key"]]
            normalized = merge_module.normalize_score(score, metric)
            rating = merge_module.convert_to_rating(normalized, metric, is_long=True)
            weight = float(merge_module.WEIGHTS_LONG.get(metric, 0.0))
            if rating is not None:
                available_weight += weight
                weighted_rating += rating * weight
            summary[metric] = {
                "raw_score": score,
                "normalized_score": normalized,
                "rating": rating,
                "weight": weight,
                "num_videos": combined["num_videos"],
                "lower_is_better": METRICS[metric]["lower_is_better"],
                "missing_shards": missing,
            }

        renorm = weighted_rating / available_weight if available_weight > 0 else None
        model_summary = {
            "dataset_name": args.dataset_name,
            "model": model,
            "metrics": summary,
            "available_weight": available_weight,
            "official_partial_weighted_rating": weighted_rating,
            "renormalized_available_rating_no_naturalness": renorm,
            "skipped_metrics": ["naturalness", "drifting_naturalness", "throughput"],
            "note": (
                "Naturalness and drifting_naturalness are skipped because no API key is used. "
                "Use renormalized_available_rating_no_naturalness for within-run comparison only; "
                "it is not directly comparable to full HeliosBench total rating."
            ),
        }
        (model_dir / "merged_results_no_naturalness.json").write_text(
            json.dumps(model_summary, indent=2), encoding="utf-8"
        )
        all_models[model] = model_summary
        flat = {
            "model": model,
            "available_weight": available_weight,
            "official_partial_weighted_rating": weighted_rating,
            "renormalized_available_rating_no_naturalness": renorm,
        }
        for metric, item in summary.items():
            if item.get("missing"):
                flat[f"{metric}_raw"] = None
                flat[f"{metric}_rating"] = None
            else:
                flat[f"{metric}_raw"] = item["raw_score"]
                flat[f"{metric}_rating"] = item["rating"]
        rows.append(flat)

    all_path = merged_root / "all_models_merged_no_naturalness.json"
    all_path.write_text(json.dumps(all_models, indent=2), encoding="utf-8")

    csv_path = merged_root / "summary_no_naturalness.csv"
    fieldnames = sorted({key for row in rows for key in row})
    front = ["model", "available_weight", "official_partial_weighted_rating", "renormalized_available_rating_no_naturalness"]
    fieldnames = front + [name for name in fieldnames if name not in front]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[heliosbench-helper] merged json : {all_path}")
    print(f"[heliosbench-helper] merged csv  : {csv_path}")


def main() -> None:
    args = parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "merge":
        merge(args)
    elif args.cmd == "run-metric":
        run_metric(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()

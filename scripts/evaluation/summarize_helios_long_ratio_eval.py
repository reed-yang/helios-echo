#!/usr/bin/env python3
"""Summarize full/start/end long-video evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
from collections import defaultdict
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--models", nargs="+", default=["base", "distilled"])
    parser.add_argument("--segments", nargs="+", default=["full", "start15", "end15"])
    parser.add_argument("--metrics", nargs="+", default=["dover", "pickscore", "hpsv3"])
    parser.add_argument("--vbench-dims", nargs="+", default=[])
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric_metric_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys = set()
    for row in rows:
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def load_regular_segments(output_root: pathlib.Path, prefix: str, segments: list[str]) -> dict[str, list[dict[str, Any]]]:
    data = {}
    for segment in segments:
        path = output_root / f"{prefix}_{segment}" / "results.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        data[segment] = rows
    return data


def model_from_run(row: dict[str, Any]) -> str:
    run_name = str(row.get("run_name", ""))
    return run_name.split("/", 1)[0] if run_name else ""


def summarize_regular(
    data: dict[str, list[dict[str, Any]]],
    models: list[str],
    segments: list[str],
) -> list[dict[str, Any]]:
    if not data:
        return []

    metric_keys = sorted({key for rows in data.values() for key in numeric_metric_keys(rows)})
    by_segment_model_metric: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)

    for segment, rows in data.items():
        for row in rows:
            model = model_from_run(row)
            if model not in models:
                continue
            video_id = str(row.get("video_id"))
            for key, value in row.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    by_segment_model_metric[(segment, model, key)][video_id] = float(value)

    out_rows: list[dict[str, Any]] = []
    for model in models:
        for key in metric_keys:
            full_values = list(by_segment_model_metric.get(("full", model, key), {}).values())
            start_map = by_segment_model_metric.get(("start15", model, key), {})
            end_map = by_segment_model_metric.get(("end15", model, key), {})
            common = sorted(set(start_map) & set(end_map))
            drift_values = [abs(start_map[vid] - end_map[vid]) for vid in common]
            delta_values = [end_map[vid] - start_map[vid] for vid in common]
            row = {
                "family": "extended_quality_preference",
                "model": model,
                "metric": key,
                "full_mean": mean(full_values),
                "start15_mean": mean(list(start_map.values())),
                "end15_mean": mean(list(end_map.values())),
                "end_minus_start_mean": mean(delta_values),
                "drift_abs_mean": mean(drift_values),
                "full_n": len(full_values),
                "drift_n": len(drift_values),
            }
            if any(row[field] is not None for field in ["full_mean", "start15_mean", "end15_mean", "drift_abs_mean"]):
                out_rows.append(row)
    return out_rows


def load_vbench_segments(
    output_root: pathlib.Path,
    prefix: str,
    segments: list[str],
    models: list[str],
    dims: list[str],
) -> dict[tuple[str, str, str], dict[str, float]]:
    data: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for segment in segments:
        for model in models:
            base = output_root / f"{prefix}_{segment}_vbench" / model
            if not base.exists():
                continue
            for path in sorted(base.glob("shard_*/*_eval_results.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for dim, result in payload.items():
                    if dims and dim not in dims:
                        continue
                    if not isinstance(result, list) or len(result) < 2:
                        continue
                    video_results = result[1]
                    if not isinstance(video_results, list):
                        continue
                    for item in video_results:
                        if not isinstance(item, dict) or "video_results" not in item:
                            continue
                        video_path = pathlib.Path(str(item.get("video_path", "")))
                        video_id = video_path.stem
                        value = item["video_results"]
                        if isinstance(value, (int, float)):
                            data[(segment, model, dim)][video_id] = float(value)
    return data


def summarize_vbench(
    data: dict[tuple[str, str, str], dict[str, float]],
    models: list[str],
    dims: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for dim in dims:
            full_map = data.get(("full", model, dim), {})
            start_map = data.get(("start15", model, dim), {})
            end_map = data.get(("end15", model, dim), {})
            common = sorted(set(start_map) & set(end_map))
            drift_values = [abs(start_map[vid] - end_map[vid]) for vid in common]
            delta_values = [end_map[vid] - start_map[vid] for vid in common]
            row = {
                "family": "vbench_custom_video",
                "model": model,
                "metric": dim,
                "full_mean": mean(list(full_map.values())),
                "start15_mean": mean(list(start_map.values())),
                "end15_mean": mean(list(end_map.values())),
                "end_minus_start_mean": mean(delta_values),
                "drift_abs_mean": mean(drift_values),
                "full_n": len(full_map),
                "drift_n": len(drift_values),
            }
            if any(row[field] is not None for field in ["full_mean", "start15_mean", "end15_mean", "drift_abs_mean"]):
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / f"{args.output_prefix}_ratio_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    regular_data = load_regular_segments(args.output_root, args.output_prefix, args.segments)
    regular_rows = summarize_regular(regular_data, args.models, args.segments)
    vbench_data = load_vbench_segments(args.output_root, args.output_prefix, args.segments, args.models, args.vbench_dims)
    vbench_rows = summarize_vbench(vbench_data, args.models, args.vbench_dims)
    all_rows = regular_rows + vbench_rows

    write_csv(out_dir / "extended_quality_preference_summary.csv", regular_rows)
    write_csv(out_dir / "vbench_custom_video_summary.csv", vbench_rows)
    write_csv(out_dir / "all_ratio_metrics_summary.csv", all_rows)
    (out_dir / "summary_manifest.json").write_text(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "output_prefix": args.output_prefix,
                "segments": args.segments,
                "models": args.models,
                "metrics": args.metrics,
                "vbench_dims": args.vbench_dims,
                "regular_rows": len(regular_rows),
                "vbench_rows": len(vbench_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ratio-summary] output: {out_dir}")
    print(f"[ratio-summary] rows  : {len(all_rows)}")


if __name__ == "__main__":
    main()

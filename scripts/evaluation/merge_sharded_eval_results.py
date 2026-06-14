#!/usr/bin/env python3
"""Merge per-shard evaluation results into one result directory."""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

from common import EvalRow
from io_utils import read_jsonl, write_all, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--shard-glob", type=str, default="{output_name}_shard_*")
    return parser.parse_args()


def sort_key(row: EvalRow) -> tuple[str, int, str]:
    idx = row.idx if row.idx is not None else 10**12
    return (row.run_name, idx, row.video_id)


def main() -> None:
    args = parse_args()
    pattern = args.shard_glob.format(output_name=args.output_name)
    shard_dirs = sorted(path for path in args.output_root.glob(pattern) if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"no shard result dirs found: {args.output_root / pattern}")

    merged: dict[tuple[str, str], EvalRow] = {}
    sources: list[str] = []
    for shard_dir in shard_dirs:
        result_path = shard_dir / "results.jsonl"
        if not result_path.exists():
            raise FileNotFoundError(f"missing shard results: {result_path}")
        sources.append(str(result_path))
        for row in read_jsonl(result_path):
            key = (row.run_name, row.video_id)
            if key not in merged:
                merged[key] = row
            else:
                merged[key].metrics.update(row.metrics)
                merged[key].errors.update(row.errors)

    rows = sorted(merged.values(), key=sort_key)
    out_dir = args.output_root / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_all(rows, out_dir)
    write_jsonl([EvalRow(**asdict(row)) for row in rows], out_dir / "manifest.jsonl")
    (out_dir / "shard_sources.json").write_text(
        json.dumps({"sources": sources}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[merge-shards] shards : {len(shard_dirs)}")
    print(f"[merge-shards] rows   : {len(rows)}")
    print(f"[merge-shards] output : {out_dir}")


if __name__ == "__main__":
    main()

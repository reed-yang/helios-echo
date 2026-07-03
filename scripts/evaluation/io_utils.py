"""JSONL/result IO for the segment-drift evaluation backend.

Reconstructed to match the callers: ``run_helios_long_segment_metric.py`` and
``merge_sharded_eval_results.py`` round-trip ``EvalRow`` via these helpers, and
the sharded launcher reads ``results.jsonl`` (rows with ``metrics``/``errors``).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from typing import List

from common import EvalRow


def read_jsonl(path) -> List[EvalRow]:
    """Load ``results.jsonl`` / ``manifest.jsonl`` back into ``EvalRow`` objects."""
    rows: List[EvalRow] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(
            EvalRow(
                video_path=d.get("video_path", ""),
                prompt=d.get("prompt", ""),
                run_name=d.get("run_name", ""),
                video_id=d.get("video_id", ""),
                idx=d.get("idx"),
                metrics=dict(d.get("metrics") or {}),
                errors=dict(d.get("errors") or {}),
            )
        )
    return rows


def write_jsonl(rows: List[EvalRow], path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_all(rows: List[EvalRow], out_dir) -> None:
    """Write the canonical ``results.jsonl`` (+ a readable ``results.json``)."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, out_dir / "results.jsonl")
    (out_dir / "results.json").write_text(
        json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

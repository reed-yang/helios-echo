#!/usr/bin/env python3
"""Aggregate vs24 eval outputs into a comparison markdown report.

Reads, per set (long | eventswitch):
  * segment-drift summary:  <ratio>/<prefix>_ratio_summary/all_ratio_metrics_summary.csv
    (DOVER / PickScore / HPSv3 + VBench dims, with full/start15/end15 + drift)
  * HeliosBench summary:    <heliosbench>/.../results_merged_no_naturalness/summary_no_naturalness.csv

Produces a markdown report comparing every checkpoint, ordered by training step,
with trend notes. Robust to partial/missing data (skips what isn't there yet).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re

METRICS_ROOT = pathlib.Path("/mnt/beegfs/xiangbo/helios_runs/eval_metrics")

# checkpoint -> training step (for ordering)
def step_of(model: str) -> int:
    m = re.search(r"(\d+)", model)
    return int(m.group(1)) if m else 0


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x, nd=4):
    if x is None or x == "" or x == "None":
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (ValueError, TypeError):
        return str(x)


def find_ratio_summary(set_name: str) -> pathlib.Path:
    return (
        METRICS_ROOT
        / f"vs24_{set_name}"
        / "ratio"
        / f"vs24_{set_name}_ratio0p15_ratio_summary"
        / "all_ratio_metrics_summary.csv"
    )


HB_METRICS = [
    "aesthetic", "motion_amplitude", "motion_smoothness", "semantic",
    "drifting_aesthetic", "drifting_motion_smoothness", "drifting_semantic",
]


def heliosbench_merged_root(set_name: str) -> pathlib.Path:
    return (
        METRICS_ROOT / f"vs24_{set_name}" / "heliosbench" / f"vs24_{set_name}"
        / "results_merged_no_naturalness"
    )


def read_heliosbench(set_name: str) -> dict[str, dict[str, float]]:
    """model -> {metric: average_score} from per-model <metric>_results.json."""
    import json

    root = heliosbench_merged_root(set_name)
    out: dict[str, dict[str, float]] = {}
    if not root.is_dir():
        return out
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        vals = {}
        for metric in HB_METRICS:
            p = model_dir / f"{metric}_results.json"
            if p.exists():
                try:
                    vals[metric] = json.loads(p.read_text())["average_score"]
                except (ValueError, KeyError):
                    pass
        if vals:
            out[model_dir.name] = vals
    return out


def pivot_ratio(rows: list[dict], metric_key: str, col: str) -> dict[str, str]:
    """model -> value for a given metric and column (e.g. full_mean)."""
    out = {}
    for r in rows:
        if r.get("metric") == metric_key:
            out[r.get("model", "")] = r.get(col, "")
    return out


LOWER_BETTER = {"motion_amplitude"}  # heliosbench: most higher=better; drift always lower=better


def trend_note(models: list[str], values: dict[str, float], lower_is_better: bool) -> str:
    pts = [(m, values[m]) for m in models if values.get(m) is not None]
    if len(pts) < 2:
        return ""
    best = (min if lower_is_better else max)(pts, key=lambda kv: kv[1])
    first, last = pts[0][1], pts[-1][1]
    delta = last - first
    direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    return f"best **{best[0]}** ({best[1]:.3f}); step0→last {direction} {delta:+.3f}"


def section_for_set(set_name: str, title: str) -> str:
    ratio_rows = read_csv(find_ratio_summary(set_name))
    helios = read_heliosbench(set_name)

    models = sorted(
        ({r.get("model", "") for r in ratio_rows} | set(helios)) - {""}, key=step_of
    )
    metrics = sorted({r.get("metric", "") for r in ratio_rows} - {""})

    lines = [f"## {title}", ""]
    if not models:
        lines += ["_No results available yet for this set._", ""]
        return "\n".join(lines)
    lines += [f"Checkpoints (by step): {', '.join(models)}", ""]

    header = "| metric | " + " | ".join(models) + " | trend |"
    sep = "|" + "---|" * (len(models) + 2)

    # --- Segment-drift: full-video score ---
    if ratio_rows:
        lines += ["### Segment-drift quality / preference (full video)", "",
                  "DOVER (no-ref quality), PickScore & HPSv3 (per-frame text-image preference), VBench dims.", "",
                  header, sep]
        for metric in metrics:
            full = {m: _f(v) for m, v in pivot_ratio(ratio_rows, metric, "full_mean").items()}
            note = trend_note(models, full, lower_is_better=False)
            lines.append(f"| {metric} | " + " | ".join(fmt(full.get(m)) for m in models) + f" | {note} |")
        lines += ["",
                  "### Temporal drift  |end15 − start15|  (lower = more stable over the clip)", "",
                  header, sep]
        for metric in metrics:
            drift = {m: _f(v) for m, v in pivot_ratio(ratio_rows, metric, "drift_abs_mean").items()}
            note = trend_note(models, drift, lower_is_better=True)
            lines.append(f"| {metric} | " + " | ".join(fmt(drift.get(m)) for m in models) + f" | {note} |")
        lines.append("")

    # --- HeliosBench ---
    if helios:
        lines += ["### HeliosBench", "", header, sep]
        for metric in HB_METRICS:
            vals = {m: helios.get(m, {}).get(metric) for m in models}
            lib = metric in LOWER_BETTER or metric.startswith("drifting_")
            note = trend_note(models, {k: v for k, v in vals.items() if v is not None}, lower_is_better=lib)
            lines.append(f"| {metric} | " + " | ".join(fmt(vals.get(m)) for m in models) + f" | {note} |")
        lines.append("")

    return "\n".join(lines)


def _f(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


METHODOLOGY = """## Methodology & how to read this

**Suite:** the merged `yushen/long-video-eval` suite — HeliosBench (aesthetic / motion / semantic + their start→end *drifting* variants), and a segment-drift backend running **DOVER** (no-reference video quality), **PickScore** & **HPSv3** (per-frame text↔image preference), and **VBench** custom-video dims (subject/background consistency, temporal flickering, motion smoothness, dynamic degree, aesthetic/imaging quality). Each segment metric is computed on three slices of every video: **full**, **start15** (first 15 %), **end15** (last 15 %).

**Checkpoints.** Each training checkpoint is a column. `step-0` is the starting point (base model before this continue-train); higher steps = more training. Long set = 7 steps × 20 videos; eventswitch set = 14 checkpoints × 40 videos.

**How to compare (important):**
- **Within a column / across columns at the same row** is the valid comparison — same metric, same segment, different checkpoints.
- The **full** table and the **drift** table answer different questions. *Full* = overall per-frame quality/preference sampled across the whole clip. *Drift* = `|end15 − start15|`, i.e. how much the metric changes from the clip's opening to its end (**lower = more temporally stable**); start15 and end15 are equal-length clips so this is apples-to-apples.
- **Do NOT compare `full` against `start15/end15` absolute values** for PickScore/HPSv3 — they are per-frame models and the slices sample different numbers/positions of frames, so absolute levels shift. Use *full* for level, *drift* for stability.
- **Length caveat:** these are 58–91 s videos. HeliosBench drift is the purpose-built long-video signal; DOVER/PickScore/HPSv3/VBench are per-frame or short-clip models read here as segment-wise / relative signals (see the length table in the working notes). VideoAlign is intentionally excluded (short-video reward model).
- **Eventswitch caveat:** these videos switch through ~6 prompts; text-alignment metrics (semantic / PickScore / HPSv3) score every frame against the **first-segment prompt only**, so they are meaningful for the opening and increasingly loose afterward. Prompt-free metrics (DOVER, VBench consistency/motion, HeliosBench aesthetic/motion) are unaffected.

"""


def key_observations(set_name: str) -> str:
    """Mechanical highlights: per-metric best checkpoint + step0→last direction."""
    ratio_rows = read_csv(find_ratio_summary(set_name))
    if not ratio_rows:
        return ""
    models = sorted({r.get("model", "") for r in ratio_rows} - {""}, key=step_of)
    if len(models) < 2:
        return ""
    notes = []
    # headline families
    for metric, lib, label in [
        ("dover_overall", False, "DOVER overall quality"),
        ("pickscore_frame_mean", False, "PickScore (prompt alignment)"),
        ("hpsv3_frame_mean", False, "HPSv3 (human preference)"),
    ]:
        full = {m: _f(v) for m, v in pivot_ratio(ratio_rows, metric, "full_mean").items()}
        pts = [(m, full[m]) for m in models if full.get(m) is not None]
        if len(pts) < 2:
            continue
        best = max(pts, key=lambda kv: kv[1])
        d = pts[-1][1] - pts[0][1]
        arrow = "improves" if d > 0 else "declines"
        notes.append(f"- **{label}:** best at `{best[0]}` ({best[1]:.3f}); overall {arrow} from step-0 to {pts[-1][0]} ({d:+.3f}).")
    if not notes:
        return ""
    return "### Key observations (auto)\n\n" + "\n".join(notes) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team/docs/VS24_EVAL_REPORT.md")
    args = ap.parse_args()

    parts = [
        "# vs24 checkpoint evaluation — long & event-switching",
        "",
        "_Auto-generated by `scripts/evaluation/make_eval_report.py` (re-run to refresh). Tables fill in as the eval pipeline completes._",
        "",
        METHODOLOGY,
        section_for_set("long", "Long single-prompt videos (`eval_out_vs24_long`, ~91 s)"),
        key_observations("long"),
        section_for_set("eventswitch", "Prompt-switching videos (`eval_out_vs24_eventswitch`, ~58 s)"),
        key_observations("eventswitch"),
    ]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

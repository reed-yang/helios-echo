#!/usr/bin/env python3
"""Drift-onset report: reproducible CLI for the r5 long-horizon saturation-drift
metrics (onset timing, recovery, runaway, motion freeze) over one or more P2
interim campaigns.

This promotes the throwaway analysis in ``/tmp/r5_drift_onset.py`` (see
``logs/research/read-r5-drift-onset-analysis.md`` for the original derivation
and published numbers) into a maintained CLI so the same drift judgements can
be recomputed on any campaign directory, of any run length, without editing
code.

Reads per-case metrics JSON files produced by
``tools/long_video_eval/scripts/run_helios_long_timeseries_metric.py`` from
``<results-root>/<campaign>/metrics/*.json``. Each file is expected to expose
``n_chunks``, ``per_chunk.saturation``, ``per_chunk.motion`` and
``slopes.saturation.{ols,theil_sen}`` (see the source report's "Schema note").

Metrics computed per case:
  1. onset (a): first chunk index after which the 60s rolling mean of
     saturation stays above ``baseline_mean + 2*baseline_std`` (baseline =
     first 60s of the same video) for at least 60s of forward persistence.
     ``None`` if it never triggers.
  2. onset (b): first chunk index at which the cumulative OLS slope over
     ``[0, t]`` turns positive and stays positive through the end of the run.
  3. recovery ratio: mean of the last 10 chunks / max of the whole run.
  4. whole-run saturation slope: read directly from the metrics JSON's own
     precomputed ``slopes.saturation.theil_sen`` field (this is what the
     source analysis compared against the recorded verdict table -- it never
     recomputed a full-series Theil-Sen slope itself, it forwarded the field
     the upstream metric script already produced). Use ``--slope-kind ols``
     to read ``slopes.saturation.ols`` instead.
  5. runaway flag: quartile means Q1..Q4 of the saturation series; runaway if
     ``Q4/Q1 > 2`` and ``Q4 >= Q3``.
  6. motion freeze flag: any run of >=15 consecutive chunks with motion below
     ``max(1e-4, 0.05 * p90(motion))``.

Short-run guard: onset (a) requires a full 60s baseline window and needs
another 60s of runway to confirm persistence; recovery ratio needs at least
10 chunks; runaway needs at least 4 chunks (one full quartile). When a
campaign's run length does not cover a window, the metric is reported as
``None`` with an explicit reason instead of raising or silently truncating.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "p2_interim"

BASELINE_SECONDS = 60.0
ROLL_SECONDS = 60.0
RECOVERY_TAIL_CHUNKS = 10
RUNAWAY_RATIO_THRESH = 2.0
FREEZE_THRESH_FRAC = 0.05
FREEZE_MIN_RUN_CHUNKS = 15

PROMPT_INDEX_RE = re.compile(r"prompt_(\d+)")


# ---- data loading -----------------------------------------------------------


class Case:
    """One (campaign, prompt) metrics file, loaded and index-tagged."""

    def __init__(self, path: Path, prompt_index: Optional[int]):
        data = json.loads(path.read_text())
        self.path = path
        self.name = path.stem
        self.prompt_index = prompt_index
        self.n_chunks = int(data["n_chunks"])
        self.sat = np.asarray(data["per_chunk"]["saturation"], dtype=float)
        self.mot = np.asarray(data["per_chunk"]["motion"], dtype=float)
        if len(self.sat) != self.n_chunks or len(self.mot) != self.n_chunks:
            raise ValueError(
                f"{path}: n_chunks={self.n_chunks} but "
                f"len(saturation)={len(self.sat)} len(motion)={len(self.mot)}"
            )
        self.slopes = data.get("slopes", {})


def discover_prompt_index(path: Path) -> Optional[int]:
    m = PROMPT_INDEX_RE.search(path.stem)
    return int(m.group(1)) if m else None


def load_campaign(results_root: Path, campaign: str) -> list:
    metrics_dir = results_root / campaign / "metrics"
    files = sorted(metrics_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no metrics JSON files under {metrics_dir}")
    return [Case(fp, discover_prompt_index(fp)) for fp in files]


# ---- shared numeric helpers --------------------------------------------------


def rolling_mean(x: np.ndarray, window_chunks: int) -> np.ndarray:
    """Trailing simple moving average ending at each index; NaN before window fills."""
    out = np.full(len(x), np.nan)
    if window_chunks <= 0:
        return out
    csum = np.concatenate([[0.0], np.cumsum(x)])
    for i in range(len(x)):
        lo = i - window_chunks + 1
        if lo < 0:
            continue
        out[i] = (csum[i + 1] - csum[lo]) / window_chunks
    return out


def theil_sen_slope(x: np.ndarray, y: np.ndarray, max_pairs: int = 4000, seed: int = 0) -> float:
    n = len(x)
    rng = np.random.default_rng(seed)
    if n * (n - 1) / 2 <= max_pairs:
        idx_i, idx_j = np.triu_indices(n, k=1)
    else:
        idx_i = rng.integers(0, n, max_pairs)
        idx_j = rng.integers(0, n, max_pairs)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
    dx = x[idx_j] - x[idx_i]
    dy = y[idx_j] - y[idx_i]
    valid = dx != 0
    slopes = dy[valid] / dx[valid]
    return float(np.median(slopes)) if len(slopes) else float("nan")


# ---- per-metric computations --------------------------------------------------


class MetricResult:
    """A metric value plus an optional short-run-guard reason (value stays None)."""

    def __init__(self, value, reason: Optional[str] = None, extra: Optional[dict] = None):
        self.value = value
        self.reason = reason
        self.extra = extra or {}


def onset_def_a(sat: np.ndarray, chunk_dt: float) -> MetricResult:
    n = len(sat)
    if n * chunk_dt < BASELINE_SECONDS:
        return MetricResult(
            None,
            reason=f"run length {n * chunk_dt:.1f}s < baseline window {BASELINE_SECONDS:.0f}s",
        )
    baseline_n = max(1, round(BASELINE_SECONDS / chunk_dt))
    baseline = sat[:baseline_n]
    thresh = baseline.mean() + 2 * baseline.std(ddof=0)
    window_n = max(1, round(ROLL_SECONDS / chunk_dt))
    if window_n > n:
        return MetricResult(
            None,
            reason=f"roll window {ROLL_SECONDS:.0f}s needs {window_n} chunks > {n} available",
        )
    roll = rolling_mean(sat, window_n)
    above = np.where(np.isnan(roll), False, roll > thresh)
    for i in range(n):
        if np.isnan(roll[i]):
            continue
        end = min(n, i + window_n)
        if end - i < window_n:
            break  # not enough remaining room to confirm 60s persistence
        if above[i:end].all():
            return MetricResult(
                i * chunk_dt,
                extra={"threshold": thresh, "baseline_mean": baseline.mean(), "baseline_std": baseline.std(ddof=0)},
            )
    return MetricResult(
        None,
        extra={"threshold": thresh, "baseline_mean": baseline.mean(), "baseline_std": baseline.std(ddof=0)},
    )


def onset_def_b(sat: np.ndarray, chunk_dt: float) -> MetricResult:
    n = len(sat)
    if n < 3:
        return MetricResult(None, reason=f"need >=3 chunks for a cumulative OLS slope, got {n}")
    x = np.arange(n) * chunk_dt
    slopes = np.full(n, np.nan)
    for i in range(2, n):
        xi = x[: i + 1]
        yi = sat[: i + 1]
        A = np.vstack([xi, np.ones_like(xi)]).T
        m, _c = np.linalg.lstsq(A, yi, rcond=None)[0]
        slopes[i] = m
    idxs = np.where(~np.isnan(slopes))[0]
    for i in idxs:
        tail = slopes[i:]
        tail_valid = tail[~np.isnan(tail)]
        if len(tail_valid) == 0:
            continue
        if (tail_valid > 0).all():
            return MetricResult(i * chunk_dt)
    return MetricResult(None)


def recovery_ratio(sat: np.ndarray) -> MetricResult:
    n = len(sat)
    if n < RECOVERY_TAIL_CHUNKS:
        return MetricResult(
            None, reason=f"need >={RECOVERY_TAIL_CHUNKS} chunks for the recovery tail, got {n}"
        )
    tail_mean = sat[-RECOVERY_TAIL_CHUNKS:].mean()
    peak = sat.max()
    if peak == 0:
        return MetricResult(None, reason="whole-run saturation peak is 0")
    return MetricResult(float(tail_mean / peak))


def quartile_means(sat: np.ndarray):
    n = len(sat)
    q = n // 4
    parts = [sat[:q], sat[q : 2 * q], sat[2 * q : 3 * q], sat[3 * q :]]
    return [float(p.mean()) if len(p) else float("nan") for p in parts]


def runaway_flag(sat: np.ndarray) -> MetricResult:
    n = len(sat)
    if n < 4:
        return MetricResult(None, reason=f"need >=4 chunks for quartile split, got {n}")
    q1, _q2, q3, q4 = quartile_means(sat)
    if q1 == 0:
        return MetricResult(None, reason="Q1 mean is 0, Q4/Q1 undefined")
    flag = (q4 / q1 > RUNAWAY_RATIO_THRESH) and (q4 >= q3)
    return MetricResult(bool(flag), extra={"q1": q1, "q3": q3, "q4": q4, "q4_over_q1": q4 / q1})


def motion_freeze_flag(mot: np.ndarray) -> MetricResult:
    n = len(mot)
    if n < FREEZE_MIN_RUN_CHUNKS:
        return MetricResult(
            None,
            reason=f"need >={FREEZE_MIN_RUN_CHUNKS} chunks for the freeze run-length scan, got {n}",
        )
    med = np.median(mot)
    p90 = np.percentile(mot, 90)
    low = mot < max(1e-4, FREEZE_THRESH_FRAC * p90)
    runs = []
    i = 0
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if j - i >= FREEZE_MIN_RUN_CHUNKS:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return MetricResult(len(runs) > 0, extra={"median_motion": float(med), "p90_motion": float(p90), "runs": runs})


def bin_trajectory(sat: np.ndarray, chunk_dt: float, bin_seconds: float) -> np.ndarray:
    n = len(sat)
    x_sec = np.arange(n) * chunk_dt
    nbins = max(1, int(np.ceil((n * chunk_dt) / bin_seconds)))
    means = np.full(nbins, np.nan)
    for b in range(nbins):
        lo, hi = b * bin_seconds, (b + 1) * bin_seconds
        mask = (x_sec >= lo) & (x_sec < hi)
        if mask.any():
            means[b] = sat[mask].mean()
    return means


# ---- per-case aggregation -----------------------------------------------------


def compute_case(case: Case, chunk_dt: float, slope_kind: str) -> dict:
    slope = case.slopes.get("saturation", {}).get(slope_kind)
    return {
        "name": case.name,
        "prompt_index": case.prompt_index,
        "n_chunks": case.n_chunks,
        "onset_a": onset_def_a(case.sat, chunk_dt),
        "onset_b": onset_def_b(case.sat, chunk_dt),
        "recovery": recovery_ratio(case.sat),
        "slope": MetricResult(slope),
        "runaway": runaway_flag(case.sat),
        "freeze": motion_freeze_flag(case.mot),
    }


def median_range(values):
    finite = [v for v in values if v is not None]
    if not finite:
        return None, None, None
    return float(np.median(finite)), float(min(finite)), float(max(finite))


# ---- rendering ------------------------------------------------------------


def fmt(v, nd=4):
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def render_campaign(campaign: str, cases: list, rows: list, baseline_rows: Optional[dict]) -> list:
    lines = [f"=== {campaign} (n_cases={len(cases)}) ==="]
    header = f"{'case':<14}{'onset_a(s)':>12}{'onset_b(s)':>12}{'recovery':>10}{'slope':>10}{'runaway':>9}{'freeze':>8}"
    if baseline_rows is not None:
        header += f"{'d_onset_a':>12}{'d_onset_b':>12}{'d_recov':>10}"
    lines.append(header)
    for row in rows:
        line = (
            f"{row['name']:<14}"
            f"{fmt(row['onset_a'].value):>12}"
            f"{fmt(row['onset_b'].value):>12}"
            f"{fmt(row['recovery'].value):>10}"
            f"{fmt(row['slope'].value):>10}"
            f"{fmt(row['runaway'].value):>9}"
            f"{fmt(row['freeze'].value):>8}"
        )
        if baseline_rows is not None:
            base = baseline_rows.get(row["prompt_index"])
            if base is None:
                line += f"{'n/a':>12}{'n/a':>12}{'n/a':>10}"
            else:
                def delta(a, b):
                    if a is None or b is None:
                        return None
                    return a - b

                line += (
                    f"{fmt(delta(row['onset_a'].value, base['onset_a'].value)):>12}"
                    f"{fmt(delta(row['onset_b'].value, base['onset_b'].value)):>12}"
                    f"{fmt(delta(row['recovery'].value, base['recovery'].value)):>10}"
                )
        lines.append(line)
        for key in ("onset_a", "onset_b", "recovery", "runaway", "freeze"):
            reason = row[key].reason
            if reason:
                lines.append(f"    ! {key} guard: {reason}")

    a_med, a_min, a_max = median_range([r["onset_a"].value for r in rows])
    b_med, b_min, b_max = median_range([r["onset_b"].value for r in rows])
    r_med, r_min, r_max = median_range([r["recovery"].value for r in rows])
    a_n = sum(1 for r in rows if r["onset_a"].value is not None)
    b_n = sum(1 for r in rows if r["onset_b"].value is not None)
    runaway_n = sum(1 for r in rows if r["runaway"].value)
    runaway_evaluable = sum(1 for r in rows if r["runaway"].value is not None)
    freeze_n = sum(1 for r in rows if r["freeze"].value)
    lines.append(
        f"{'SUMMARY':<14}"
        f"onset_a median={fmt(a_med)} range=[{fmt(a_min)},{fmt(a_max)}] n={a_n}/{len(rows)}  "
        f"onset_b median={fmt(b_med)} range=[{fmt(b_min)},{fmt(b_max)}] n={b_n}/{len(rows)}  "
        f"recovery median={fmt(r_med)} range=[{fmt(r_min)},{fmt(r_max)}]  "
        f"runaway={runaway_n}/{runaway_evaluable}  freeze={freeze_n}/{len(rows)}"
    )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--campaign",
        action="append",
        required=True,
        dest="campaigns",
        metavar="RUN_ID",
        help="campaign directory name under --results-root; repeatable",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="root containing <campaign>/metrics/*.json (default: results/p2_interim)",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="video fps used to convert chunks to seconds")
    parser.add_argument(
        "--chunk-frames", type=int, default=33, help="RGB frames per chunk (chunk_dt = chunk_frames / fps)"
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="RUN_ID",
        help="campaign to diff against per-case, matched by prompt_index (must be in --campaign list or loadable separately)",
    )
    parser.add_argument("--json", default=None, metavar="OUT", help="write a machine-readable JSON dump to this path")
    parser.add_argument("--bin-seconds", type=float, default=30.0, help="bin width for the saturation trajectory")
    parser.add_argument(
        "--slope-kind",
        choices=("theil_sen", "ols"),
        default="theil_sen",
        help="which precomputed whole-run slopes.saturation.<kind> field to report (default: theil_sen, "
        "matching the source analysis's cross-check against the recorded verdict table)",
    )
    args = parser.parse_args(argv)

    results_root = Path(args.results_root)
    chunk_dt = args.chunk_frames / args.fps

    campaign_cases = {}
    campaign_rows = {}
    for campaign in args.campaigns:
        cases = load_campaign(results_root, campaign)
        campaign_cases[campaign] = cases
        campaign_rows[campaign] = [compute_case(c, chunk_dt, args.slope_kind) for c in cases]

    baseline_by_index = None
    if args.baseline:
        if args.baseline not in campaign_cases:
            base_cases = load_campaign(results_root, args.baseline)
            base_rows = [compute_case(c, chunk_dt, args.slope_kind) for c in base_cases]
        else:
            base_rows = campaign_rows[args.baseline]
        baseline_by_index = {r["prompt_index"]: r for r in base_rows}

    out_lines = []
    json_dump = {"chunk_dt_seconds": chunk_dt, "campaigns": {}}
    for campaign in args.campaigns:
        rows = campaign_rows[campaign]
        base = baseline_by_index if (args.baseline and campaign != args.baseline) else None
        out_lines.extend(render_campaign(campaign, campaign_cases[campaign], rows, base))
        out_lines.append("")
        json_dump["campaigns"][campaign] = [
            {
                "name": r["name"],
                "prompt_index": r["prompt_index"],
                "n_chunks": r["n_chunks"],
                "onset_a": r["onset_a"].value,
                "onset_a_reason": r["onset_a"].reason,
                "onset_b": r["onset_b"].value,
                "onset_b_reason": r["onset_b"].reason,
                "recovery": r["recovery"].value,
                "recovery_reason": r["recovery"].reason,
                "slope": r["slope"].value,
                "runaway": r["runaway"].value,
                "runaway_reason": r["runaway"].reason,
                "freeze": r["freeze"].value,
                "freeze_reason": r["freeze"].reason,
            }
            for r in rows
        ]

    out_lines.append(f"--- {args.bin_seconds:.0f}s-binned mean saturation trajectory per campaign ---")
    for campaign in args.campaigns:
        cases = campaign_cases[campaign]
        binned = [bin_trajectory(c.sat, chunk_dt, args.bin_seconds) for c in cases]
        maxlen = max(len(b) for b in binned)
        padded = np.full((len(binned), maxlen), np.nan)
        for i, b in enumerate(binned):
            padded[i, : len(b)] = b
        arm_mean = np.nanmean(padded, axis=0)
        out_lines.append(f"{campaign}: nbins={maxlen}")
        out_lines.append("  " + " ".join(f"{v:.3g}" for v in arm_mean))
        json_dump["campaigns"][campaign + "__trajectory"] = arm_mean.tolist()

    print("\n".join(out_lines))

    if args.json:
        Path(args.json).write_text(json.dumps(json_dump, indent=2))
        print(f"\nwrote JSON dump to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

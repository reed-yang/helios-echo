#!/usr/bin/env python3
"""Frame-level flicker report: is there a step at every chunk boundary?

The long-video timeseries metric emits one value per 33-frame chunk, which
averages away structure at the chunk cadence. A human watching 8-minute
rollouts reported a slight flicker roughly every 1-2 seconds with the history
projection enabled; one chunk is 33 frames at 24fps = 1.375s, so the claim is
testable only per frame.

For each video this measures the per-frame mean saturation and brightness, then
asks whether the frame-to-frame jumps concentrate at frame indices that are
multiples of the chunk length. The phase control matters more than the ratio
itself: a ratio above 1 at the true boundary phase means nothing unless that
phase also ranks high among all possible phases.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results/p2_interim"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", default=[], help="run id under --results-root (repeatable)")
    parser.add_argument("--video", action="append", default=[], type=Path, help="explicit video path (repeatable)")
    parser.add_argument("--cases", default="", help="comma list of prompt indices to keep, e.g. 2,0")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--chunk-frames", type=int, default=33)
    parser.add_argument("--max-frames", type=int, default=3300, help="frames analysed per video (0 = all)")
    parser.add_argument("--width", type=int, default=160, help="analysis width; frames are downscaled for speed")
    parser.add_argument("--json", type=Path, help="write the full result table here")
    return parser.parse_args(argv)


def collect_videos(args):
    videos = list(args.video)
    keep = {int(c) for c in args.cases.split(",") if c.strip()} if args.cases else None
    for run_id in args.campaign:
        for path in sorted((args.results_root / run_id).glob("*/prompt_*.mp4")):
            index = int(path.stem.split("_")[-1])
            if keep is None or index in keep:
                videos.append(path)
    return videos


def frame_series(path, max_frames, width):
    """Per-frame mean saturation and brightness of a downscaled video."""
    capture = cv2.VideoCapture(str(path))
    saturation, brightness = [], []
    while max_frames <= 0 or len(saturation) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        height = max(1, int(frame.shape[0] * width / frame.shape[1]))
        small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        saturation.append(float(hsv[:, :, 1].mean()))
        brightness.append(float(hsv[:, :, 2].mean()))
    capture.release()
    return np.asarray(saturation), np.asarray(brightness)


def phase_profile(deltas, period):
    """Mean |delta| for each phase, where phase p collects indices i with
    i % period == p. Element i of the delta series is the step from frame i to
    i+1, so a jump entering frame b shows up at index b-1: the chunk-boundary
    phase is period-1, not 0. Rather than rely on that, the caller reports the
    empirically strongest phase and whether it is the same across videos."""
    return np.asarray([deltas[p::period].mean() if deltas[p::period].size else np.nan for p in range(period)])


def boundary_stats(series, period):
    deltas = np.abs(np.diff(series))
    if deltas.size < 3 * period:
        return None
    profile = phase_profile(deltas, period)
    boundary_phase = period - 1
    boundary = profile[boundary_phase]
    ratio = float(boundary / np.delete(profile, boundary_phase).mean())
    rank = int((profile >= boundary).sum())
    top_phase = int(np.nanargmax(profile))
    top_ratio = float(profile[top_phase] / np.delete(profile, top_phase).mean())
    spectrum = np.abs(np.fft.rfft(deltas - deltas.mean()))
    freqs = np.fft.rfftfreq(deltas.size)
    target = np.argmin(np.abs(freqs - 1.0 / period))
    band = spectrum[(freqs > 1.0 / 100) & (freqs < 1.0 / 5)]
    fft_ratio = float(spectrum[target] / np.median(band)) if band.size else float("nan")
    return {
        "boundary_ratio": ratio,
        "phase_rank": rank,
        "top_phase": top_phase,
        "top_ratio": top_ratio,
        "phases": period,
        "fft_at_period": fft_ratio,
        "mean_abs_delta": float(deltas.mean()),
        "boundary_mean_abs_delta": float(boundary),
    }


def main(argv=None):
    args = parse_args(argv)
    videos = collect_videos(args)
    if not videos:
        raise SystemExit("no videos matched")
    rows = []
    for path in videos:
        saturation, brightness = frame_series(path, args.max_frames, args.width)
        row = {
            "video": str(path.relative_to(args.results_root)) if str(path).startswith(str(args.results_root)) else str(path),
            "frames": int(saturation.size),
        }
        for name, series in (("saturation", saturation), ("brightness", brightness)):
            stats = boundary_stats(series, args.chunk_frames)
            row[name] = stats
            if stats is not None:
                print(
                    f"FLICKER video={row['video']} metric={name} "
                    f"boundary_ratio={stats['boundary_ratio']:.3f} "
                    f"phase_rank={stats['phase_rank']}/{stats['phases']} "
                    f"top_phase={stats['top_phase']}(x{stats['top_ratio']:.2f}) "
                    f"fft33={stats['fft_at_period']:.2f} frames={row['frames']}",
                    flush=True,
                )
        rows.append(row)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2))
    print("EXIT_CODE=0", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute chunk-aligned drift time series for one or more videos.

A single input writes one per-video result object. Multiple inputs write
``{"results": [<per-video result>, ...]}`` to the requested JSON path.
Incomplete trailing chunks are ignored so all reported series are aligned to
``--chunk_frames``. Motion frames are resized to a fixed resolution before
Farneback flow; saturation, chroma, and boundary metrics use native frames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from scipy.stats import theilslopes

FrameDistance = Callable[[np.ndarray, np.ndarray], float]


# Vendored from eval/1_get_motion_amplitude.py:34-59,62-69 because that script
# imports project-specific data utilities and executes a dataset-level workflow.
def compute_farneback_optical_flow(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Compute dense optical flow using the existing Farneback parameters."""
    if len(frames) < 2:
        return []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    flow_maps = []
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow_map = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow_maps.append(flow_map)
        prev_gray = gray
    return flow_maps


def _motion_score(flow_maps: Sequence[np.ndarray], downscale_size: int = 16) -> float:
    """Apply the existing downscale-and-mean motion scoring procedure."""
    if not flow_maps:
        return 0.0
    downscaled = []
    for flow in flow_maps:
        height, width = flow.shape[:2]
        new_height = max(1, int(height * (downscale_size / width)))
        downscaled.append(
            cv2.resize(flow, (downscale_size, new_height), interpolation=cv2.INTER_AREA)
        )
    average_map = np.mean(np.asarray(downscaled), axis=0)
    return abs(float(np.mean(average_map)))


def _l2_distance(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Return normalized RGB mean squared error."""
    difference = frame_a.astype(np.float32) / 255.0 - frame_b.astype(np.float32) / 255.0
    return float(np.mean(np.square(difference), dtype=np.float64))


def _select_distance_backend(use_lpips: bool) -> tuple[str, FrameDistance]:
    if not use_lpips:
        return "l2", _l2_distance

    try:
        import lpips  # type: ignore[import-not-found]
        import torch

        model = lpips.LPIPS(net="alex")
        model.eval()

        def lpips_distance(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
            arrays = []
            for frame in (frame_a, frame_b):
                tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
                arrays.append(tensor.unsqueeze(0).float().div(127.5).sub(1.0))
            with torch.no_grad():
                return float(model(arrays[0], arrays[1]).reshape(-1).mean().item())

        return "lpips", lpips_distance
    except Exception as exc:  # Import and model initialization are both optional.
        warnings.warn(
            f"LPIPS requested but unavailable ({exc}); falling back to normalized RGB MSE.",
            RuntimeWarning,
        )
        return "l2", _l2_distance


def _chunk_metrics(
    frames_bgr: Sequence[np.ndarray], motion_height: int, motion_width: int
) -> tuple[float, float, float]:
    motion_frames_rgb = []
    for frame in frames_bgr:
        height, width = frame.shape[:2]
        interpolation = (
            cv2.INTER_AREA
            if motion_height < height or motion_width < width
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            frame,
            (motion_width, motion_height),
            interpolation=interpolation,
        )
        motion_frames_rgb.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    motion = _motion_score(compute_farneback_optical_flow(motion_frames_rgb))

    saturation_sum = 0.0
    chroma_sum = 0.0
    pixel_count = 0
    for frame in frames_bgr:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_star = lab[:, :, 1] - 128.0
        b_star = lab[:, :, 2] - 128.0
        saturation_sum += float(np.sum(hsv[:, :, 1], dtype=np.float64))
        chroma_sum += float(np.sum(np.hypot(a_star, b_star), dtype=np.float64))
        pixel_count += frame.shape[0] * frame.shape[1]

    return motion, saturation_sum / pixel_count, chroma_sum / pixel_count


def _slope(values: Sequence[float]) -> dict[str, float]:
    if len(values) < 2:
        return {"ols": 0.0, "theil_sen": 0.0}
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    ols = float(np.polyfit(x, y, deg=1)[0])
    robust = float(theilslopes(y, x).slope)
    return {"ols": ols, "theil_sen": robust}


def _auc(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.trapz(np.asarray(values, dtype=np.float64), dx=1.0))


def _read_full_chunk(capture: cv2.VideoCapture, chunk_frames: int) -> list[np.ndarray]:
    frames = []
    for _ in range(chunk_frames):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def analyze_video(
    video_path: str | Path,
    chunk_frames: int = 33,
    use_lpips: bool = False,
    motion_height: int = 384,
    motion_width: int = 640,
) -> dict[str, object]:
    """Analyze complete chunks from one video and return a JSON-ready object."""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if motion_height <= 0 or motion_width <= 0:
        raise ValueError("motion_height and motion_width must be positive")

    path = Path(video_path).expanduser()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    distance_backend, frame_distance = _select_distance_backend(use_lpips)
    motion: list[float] = []
    saturation: list[float] = []
    chroma: list[float] = []
    boundary_excess: list[float] = []
    previous_last_rgb: np.ndarray | None = None
    previous_intra_baseline: float | None = None

    try:
        while True:
            frames_bgr = _read_full_chunk(capture, chunk_frames)
            if len(frames_bgr) < chunk_frames:
                break

            chunk_motion, chunk_saturation, chunk_chroma = _chunk_metrics(
                frames_bgr, motion_height, motion_width
            )
            motion.append(chunk_motion)
            saturation.append(chunk_saturation)
            chroma.append(chunk_chroma)

            frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
            if previous_last_rgb is not None and previous_intra_baseline is not None:
                boundary_excess.append(
                    frame_distance(previous_last_rgb, frames_rgb[0]) - previous_intra_baseline
                )
            adjacent_distances = [
                frame_distance(frame_a, frame_b)
                for frame_a, frame_b in zip(frames_rgb, frames_rgb[1:])
            ]
            previous_intra_baseline = (
                float(np.mean(adjacent_distances)) if adjacent_distances else 0.0
            )
            previous_last_rgb = frames_rgb[-1]
    finally:
        capture.release()

    if not motion:
        raise ValueError(
            f"Video has no complete {chunk_frames}-frame chunks: {path}"
        )

    series = {
        "motion": motion,
        "saturation": saturation,
        "chroma": chroma,
        "boundary_excess": boundary_excess,
    }
    for metric_values in series.values():
        if not all(math.isfinite(value) for value in metric_values):
            raise ValueError(f"Non-finite metric produced for video: {path}")

    return {
        "video": str(path),
        "n_chunks": len(motion),
        "chunk_frames": chunk_frames,
        "motion_resolution": [motion_height, motion_width],
        "distance_backend": distance_backend,
        "per_chunk": {
            "motion": motion,
            "saturation": saturation,
            "chroma": chroma,
        },
        "boundary_excess": boundary_excess,
        "slopes": {name: _slope(values) for name, values in series.items()},
        "auc": {
            "saturation": _auc(saturation),
            "chroma": _auc(chroma),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute chunk-aligned long-video drift time series."
    )
    parser.add_argument("videos", nargs="+", help="One or more input video paths")
    parser.add_argument("--chunk_frames", type=int, default=33)
    parser.add_argument("--motion_height", type=int, default=384)
    parser.add_argument("--motion_width", type=int, default=640)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--event_spec",
        help="Accepted for forward compatibility; event metrics are deferred",
    )
    parser.add_argument("--use_lpips", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.event_spec:
        print(
            "Notice: --event_spec is accepted but event metrics are deferred; ignoring it.",
            file=sys.stderr,
        )

    results = [
        analyze_video(
            path,
            chunk_frames=args.chunk_frames,
            use_lpips=args.use_lpips,
            motion_height=args.motion_height,
            motion_width=args.motion_width,
        )
        for path in args.videos
    ]
    payload: dict[str, object] = results[0] if len(results) == 1 else {"results": results}
    output_path = Path(args.out).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

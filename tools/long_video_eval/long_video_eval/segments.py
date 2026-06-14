"""Video segment preparation for long-video drift evaluation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .config import RuntimeConfig


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def prepare_ratio_segments(
    runtime: RuntimeConfig,
    *,
    segment_root: Path,
    ratio: float = 0.15,
    overwrite: bool = False,
) -> dict:
    """Create full/start/end segment files.

    The implementation uses stream copy for full videos and ffmpeg trimming for
    start/end. It is intentionally backend-agnostic and does not depend on
    Helios code.
    """

    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to prepare start/end video segments")

    segment_root.mkdir(parents=True, exist_ok=True)
    reports = []
    duration_seconds = runtime.duration_frames / 24.0
    # This suite assumes 24 fps by default; callers may override the metadata
    # through config-level command templates for metrics that need exact fps.
    clip_seconds = max(duration_seconds * ratio, 0.1)
    end_start = max(duration_seconds - clip_seconds, 0.0)

    for model in runtime.models:
        src_dir = runtime.source_root / model / "videos"
        for segment in ["full", "start15", "end15"]:
            dst_dir = segment_root / segment / model / "videos"
            dst_dir.mkdir(parents=True, exist_ok=True)

        for src in sorted(src_dir.glob("*.mp4")):
            full_dst = segment_root / "full" / model / "videos" / src.name
            start_dst = segment_root / "start15" / model / "videos" / src.name
            end_dst = segment_root / "end15" / model / "videos" / src.name

            if overwrite or not full_dst.exists():
                if full_dst.exists() or full_dst.is_symlink():
                    full_dst.unlink()
                full_dst.symlink_to(src.resolve())

            if overwrite or not start_dst.exists():
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(src), "-t", f"{clip_seconds:.6f}", "-c", "copy", str(start_dst)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            if overwrite or not end_dst.exists():
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        f"{end_start:.6f}",
                        "-i",
                        str(src),
                        "-t",
                        f"{clip_seconds:.6f}",
                        "-c",
                        "copy",
                        str(end_dst),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        reports.append({"model": model, "source": str(src_dir), "segment_root": str(segment_root)})

    report = {
        "segment_root": str(segment_root),
        "ratio": ratio,
        "duration_seconds": duration_seconds,
        "clip_seconds": clip_seconds,
        "reports": reports,
    }
    (segment_root / "segment_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

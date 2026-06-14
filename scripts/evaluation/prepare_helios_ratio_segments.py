#!/usr/bin/env python3
"""Prepare full/start/end ratio segments for long-video evaluation."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--prompt-file", type=pathlib.Path, required=True)
    parser.add_argument("--models", type=str, default="base,distilled")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(src: pathlib.Path, dst: pathlib.Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def copy_scores(src_video_dir: pathlib.Path, dst_video_dir: pathlib.Path) -> None:
    scores = src_video_dir / "scores.json"
    if scores.exists():
        shutil.copy2(scores, dst_video_dir / "scores.json")


def write_vbench_prompts(video_dir: pathlib.Path, videos: list[pathlib.Path], prompts: list[str]) -> None:
    prompt_map = {}
    for src in videos:
        try:
            idx = int(src.stem)
        except ValueError:
            continue
        if 0 <= idx < len(prompts):
            prompt_map[src.name] = prompts[idx]
    (video_dir / "vbench_prompts.json").write_text(
        json.dumps(prompt_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def segment_bounds(total: int, name: str, ratio: float) -> tuple[int, int]:
    # Match HeliosBench drifting scripts: int(total_frames * DRIFT_RATIO).
    n = max(1, int(total * ratio))
    n = min(n, total)
    if name == "start15":
        return 0, n
    if name == "end15":
        return total - n, total
    raise ValueError(name)


def write_segment(src: pathlib.Path, dst: pathlib.Path, name: str, fps: int, ratio: float, overwrite: bool) -> int:
    if dst.exists() and not overwrite:
        return 0

    import decord
    from diffusers.utils import export_to_video

    reader = decord.VideoReader(str(src))
    total = len(reader)
    if total <= 0:
        raise RuntimeError(f"no frames found in {src}")

    start, stop = segment_bounds(total, name, ratio)
    frames = [reader[i].asnumpy() for i in range(start, stop)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    export_to_video(frames, str(dst), fps=fps)
    return len(frames)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ratio < 0.5:
        raise ValueError(f"--ratio must be in (0, 0.5), got {args.ratio}")

    models = [name.strip() for name in args.models.replace(",", " ").split() if name.strip()]
    prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.prompt_file, args.output_root / args.prompt_file.name)

    manifest = {
        "source_root": str(args.source_root),
        "prompt_file": str(args.prompt_file),
        "fps": args.fps,
        "ratio": args.ratio,
        "segments": {
            "full": "symlink/copy of the full generated video",
            "start15": f"first {args.ratio:.0%} of frames",
            "end15": f"last {args.ratio:.0%} of frames",
        },
        "models": models,
    }

    for model in models:
        src_video_dir = args.source_root / model / "videos"
        if not src_video_dir.exists():
            raise FileNotFoundError(f"missing source video dir: {src_video_dir}")
        videos = sorted(src_video_dir.glob("*.mp4"))
        if args.max_videos > 0:
            videos = videos[: args.max_videos]

        full_dir = args.output_root / "full" / model / "videos"
        full_dir.mkdir(parents=True, exist_ok=True)
        copy_scores(src_video_dir, full_dir)
        write_vbench_prompts(full_dir, videos, prompts)
        for src in videos:
            link_or_copy(src, full_dir / src.name, args.overwrite)
        print(f"[ratio-segments] full/{model}: {len(videos)} videos -> {full_dir}")

        for segment_name in ("start15", "end15"):
            out_dir = args.output_root / segment_name / model / "videos"
            out_dir.mkdir(parents=True, exist_ok=True)
            copy_scores(src_video_dir, out_dir)
            write_vbench_prompts(out_dir, videos, prompts)
            frame_counts = []
            for src in videos:
                frame_counts.append(write_segment(src, out_dir / src.name, segment_name, args.fps, args.ratio, args.overwrite))
            nonzero = [count for count in frame_counts if count]
            frames_note = f", wrote {min(nonzero)}-{max(nonzero)} frames" if nonzero else ""
            print(f"[ratio-segments] {segment_name}/{model}: {len(videos)} videos -> {out_dir}{frames_note}")

    (args.output_root / "segments_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

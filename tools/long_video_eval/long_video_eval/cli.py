"""Command line interface for long_video_eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import build_runtime
from .manifest import build_manifest, write_manifest
from .resources import ResourceResolver
from .runner import run_suite


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--duration-frames", type=int, default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--cache-dir", default=None)


def build_from_args(args: argparse.Namespace):
    return build_runtime(
        args.config,
        source_root=args.source_root,
        prompt_file=args.prompt_file,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        duration_frames=args.duration_frames,
        models=args.models,
        gpus=args.gpus,
        cache_dir=args.cache_dir,
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    runtime = build_from_args(args)
    ok = True

    try:
        manifest = build_manifest(runtime, expected_videos=args.expected_videos, strict=args.strict)
        print("[doctor] input manifest ok")
        print(json.dumps({k: manifest[k] for k in ["dataset_name", "prompt_count", "item_count", "missing_count"]}, indent=2))
    except Exception as exc:
        ok = False
        print(f"[doctor] input manifest failed: {exc}", file=sys.stderr)

    resolver = ResourceResolver(runtime)
    for result in resolver.check_all():
        status = "OK" if result.ok else "MISSING"
        print(f"[doctor] {status:7s} {result.kind:15s} {result.id:28s} {result.message}")
        if not result.ok:
            ok = False
    return 0 if ok else 2


def cmd_setup(args: argparse.Namespace) -> int:
    runtime = build_from_args(args)
    resolver = ResourceResolver(runtime)
    resolver.setup(yes=args.yes, dry_run=args.dry_run)
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    runtime = build_from_args(args)
    manifest = build_manifest(runtime, expected_videos=args.expected_videos, strict=args.strict)
    path = write_manifest(manifest, runtime.output_root, runtime.dataset_name)
    print(f"[prepare] manifest={path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    runtime = build_from_args(args)
    resolver = ResourceResolver(runtime)
    missing = [result for result in resolver.check_all() if not result.ok]
    if missing and not args.allow_missing_resources:
        for result in missing:
            print(f"[run] missing {result.kind} {result.id}: {result.message}", file=sys.stderr)
        print("[run] run `doctor` and `setup` first, or pass --allow-missing-resources for command dry-runs", file=sys.stderr)
        return 2
    run_suite(runtime, dry_run=args.dry_run, strict=args.strict, expected_videos=args.expected_videos)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="long-video-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    doctor = sub.add_parser("doctor", help="check inputs, Python imports, code paths, and checkpoints")
    add_runtime_args(doctor)
    doctor.add_argument("--strict", action="store_true", help="fail if any indexed video is missing")
    doctor.add_argument("--expected-videos", type=int, default=0)
    doctor.set_defaults(func=cmd_doctor)

    setup = sub.add_parser("setup", help="clone/download configured resources")
    add_runtime_args(setup)
    setup.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    setup.add_argument("--dry-run", action="store_true")
    setup.set_defaults(func=cmd_setup)

    prepare = sub.add_parser("prepare", help="write an input manifest only")
    add_runtime_args(prepare)
    prepare.add_argument("--strict", action="store_true")
    prepare.add_argument("--expected-videos", type=int, default=0)
    prepare.set_defaults(func=cmd_prepare)

    run = sub.add_parser("run", help="run configured evaluation steps")
    add_runtime_args(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--strict", action="store_true")
    run.add_argument("--expected-videos", type=int, default=0)
    run.add_argument("--allow-missing-resources", action="store_true")
    run.set_defaults(func=cmd_run)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(args.func(args))

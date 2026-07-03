#!/usr/bin/env python3
"""Normalize Helios eval-output dirs into the long_video_eval input contract.

The long_video_eval suite (HeliosBench + segment-drift) expects, per "model":

    SOURCE_ROOT/<model>/videos/NNNN.mp4     # 0-based, zero-padded integer stem
    prompt.txt                              # line i (0-based) -> NNNN==i

Our raw eval outputs do not satisfy this:

  * eval_out_vs24_eventswitch/<ckpt>/1000.mp4 .. 1093.mp4   (prompt-switching grid)
  * eval_out_vs24_long/<step>/test/<id>_2178_ori2178.mp4    (id 0..19)

This script builds a normalized symlink view (no video copies) for each set and
writes one line-aligned prompt.txt per set. Each checkpoint/step becomes a
"model" so the suite can compare them side-by-side in one run.

eventswitch text prompt = the FIRST segment prompt (prompt_index==0) of each id
(per the user's choice), since these are prompt-switching videos.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
WANDER_CSV = REPO / "example" / "prompt_interactive_helios_wander.csv"

EVENTSWITCH_SRC = pathlib.Path("/mnt/beegfs/xiangbo/helios_runs/eval_out_vs24_eventswitch")
LONG_SRC = pathlib.Path("/mnt/beegfs/xiangbo/helios_runs/eval_out_vs24_long")
NORM_ROOT = pathlib.Path("/mnt/beegfs/xiangbo/helios_runs/eval_norm")

LONG_CANONICAL_RE = re.compile(r"^(\d+)_\d+_ori\d+$")  # e.g. 7_2178_ori2178


def _link(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def first_segment_prompts() -> dict[int, str]:
    """Map eventswitch video id -> the prompt_index==0 (opening) prompt."""
    out: dict[int, str] = {}
    with WANDER_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["prompt_index"]) == 0:
                out[int(row["id"])] = row["prompt"].strip()
    return out


def normalize_eventswitch() -> None:
    prompts = first_segment_prompts()
    ckpts = sorted(
        [p for p in EVENTSWITCH_SRC.iterdir() if p.is_dir()],
        key=lambda p: (p.name != "step-0", _ckpt_sort_key(p.name)),
    )
    # video id ordering is shared across all checkpoints (identical filenames)
    ref_ids = sorted(int(v.stem) for v in next(iter(ckpts)).glob("*.mp4"))
    dest_root = NORM_ROOT / "eventswitch"
    for ckpt in ckpts:
        ids = sorted(int(v.stem) for v in ckpt.glob("*.mp4"))
        if ids != ref_ids:
            print(f"WARN: {ckpt.name} ids differ from reference set", file=sys.stderr)
        vdir = dest_root / ckpt.name / "videos"
        for norm_idx, vid in enumerate(ids):
            _link(ckpt / f"{vid}.mp4", vdir / f"{norm_idx:04d}.mp4")
        print(f"[eventswitch] {ckpt.name}: {len(ids)} videos -> {vdir}")
    prompt_lines = [prompts[vid] for vid in ref_ids]
    (dest_root / "prompt.txt").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    print(f"[eventswitch] prompt.txt: {len(prompt_lines)} lines (first-segment prompts)")
    print(f"[eventswitch] models: {[c.name for c in ckpts]}")


def normalize_long() -> None:
    steps = sorted(
        [p for p in LONG_SRC.iterdir() if p.is_dir() and p.name.startswith("step-")],
        key=lambda p: int(p.name.split("-")[1]),
    )
    dest_root = NORM_ROOT / "long"
    # shared 20-prompt set
    csv_path = next(s for s in steps if (s / "test" / "_input.csv").exists()) / "test" / "_input.csv"
    id2prompt: dict[int, str] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id2prompt[int(row["id"])] = row["prompt"].strip()
    max_id = max(id2prompt)
    for step in steps:
        test_dir = step / "test"
        if not test_dir.is_dir():
            print(f"WARN: {step.name} has no test/ dir, skipping", file=sys.stderr)
            continue
        vdir = dest_root / step.name / "videos"
        n = 0
        for vid in sorted(test_dir.glob("*.mp4")):
            m = LONG_CANONICAL_RE.match(vid.stem)
            if not m:
                continue  # skip partial bare "N.mp4" reruns
            idx = int(m.group(1))
            _link(vid, vdir / f"{idx:04d}.mp4")
            n += 1
        print(f"[long] {step.name}: {n} canonical videos -> {vdir}")
    prompt_lines = [id2prompt[i] for i in range(max_id + 1)]
    (dest_root / "prompt.txt").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
    print(f"[long] prompt.txt: {len(prompt_lines)} lines")
    print(f"[long] models: {[s.name for s in steps]}")


def _ckpt_sort_key(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=["eventswitch", "long", "both"], default="both")
    args = ap.parse_args()
    if args.set in ("eventswitch", "both"):
        normalize_eventswitch()
    if args.set in ("long", "both"):
        normalize_long()


if __name__ == "__main__":
    main()

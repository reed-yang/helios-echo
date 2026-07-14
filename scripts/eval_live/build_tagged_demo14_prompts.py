#!/usr/bin/env python3
"""Build tagged variants of the demo14 eval prompt CSVs (rwtag run).

Prepends the training-time dataset tag ("<music_performance*>\n" etc.) to
every prompt of the scenes that clearly map to a v2 training dataset. Scenes
without a confident mapping (05 woman/phone, 08 AR solar system, 09, 10
magnifier, 11 turbine) get no tagged variant. Output CSVs keep the same
scene ids so infer_helios names the mp4s identically; the eval watcher
routes them to a separate demo14tag output folder.
"""
import csv
import os
import sys

# default: original demo14; pass e.g. "demo14_aligned" to tag the aligned set
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "demo14"
SRC = f"/mnt/beegfs/xiangbo/helios_runs/eval_prompts_{VARIANT}"
DST = f"{SRC}_tag"

SCENE_TAG = {
    "scene_01": "<music_performance*>",   # concert stage / band
    "scene_02": "<genesis*>",             # God figure / creation
    "scene_03": "<jesus*>",
    "scene_03a": "<jesus*>",
    "scene_03b": "<jesus*>",
    "scene_03c": "<jesus*>",
    "scene_04": "<hatching*>",            # egg wiggles / chick hatches
    "scene_06": "<realsource*>",          # robot-arm egocentric manipulation
    "scene_07": "<realsource*>",          # gray mechanical arm / cube
}

os.makedirs(DST, exist_ok=True)
for scene, tag in SCENE_TAG.items():
    src = os.path.join(SRC, f"{scene}.csv")
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["prompt"] = f"{tag}\n{r['prompt']}"
    with open(os.path.join(DST, f"{scene}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "prompt_index", "prompt"])
        w.writeheader()
        w.writerows(rows)
    print(f"{scene}: {len(rows)} prompts tagged with {tag}")
print(f"done -> {DST}")

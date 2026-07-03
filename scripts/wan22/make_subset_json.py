#!/usr/bin/env python
"""Carve a PoC / smoke subset out of an offload-format manifest (list of clip dicts).

Keeps only clips with num_frames >= MIN_FRAMES (the stage-1 dataloader drops <121), then takes
the first N entries (the corpus is ordered by content-hash id, so the head is effectively random).

  python scripts/wan22/make_subset_json.py \
      --src /mnt/beegfs/dataset/video_single_24FPS/offload_cfr_int.json \
      --out /mnt/beegfs/dataset/video_single_24FPS/offload_cfr_int_wan22_poc.json \
      --n 25000 --min-frames 121
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--min-frames", type=int, default=121)
args = ap.parse_args()

data = json.load(open(args.src))
assert isinstance(data, list), f"expected a list manifest, got {type(data)}"
kept = [d for d in data if int(d.get("num_frames", 0)) >= args.min_frames]
subset = kept[: args.n]
json.dump(subset, open(args.out, "w"))
print(
    f"src={len(data)} clips | >= {args.min_frames}f: {len(kept)} | wrote {len(subset)} -> {args.out}",
    flush=True,
)

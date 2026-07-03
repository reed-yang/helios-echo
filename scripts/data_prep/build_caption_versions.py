#!/usr/bin/env python3
"""Build a caption-versions lookup for the video_single_24FPS encode.

The offload JSON (train_24fps.json / test_24fps.json) carries two inline caption
versions per clip: cap = [short, long]. The latent encoder
(tools/offload_data/get_short-latents.py) writes one prompt embed per *version*
when given a --rewritten_json lookup of the form

    { "<bare_filename>.mp4": {"short": <text>, "long": <text>}, ... }

(the key matches the encoder's _clip_key(uttid) == "<basename>.mp4" == the JSON
"path" field). This script materializes that lookup so we can reuse the existing,
proven encoder mechanism with no invasive change. The training dataloader already
mixes "short"/"long" automatically (they are in its default caption_versions list).

Usage:
    python build_caption_versions.py \
        --in  /mnt/beegfs/dataset/video_single_24FPS/train_24fps.json \
               /mnt/beegfs/dataset/video_single_24FPS/test_24fps.json \
        --out /mnt/beegfs/dataset/video_single_24FPS/caption_versions.json
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="offload-format JSON list file(s)")
    ap.add_argument("--out", required=True, help="output lookup JSON")
    args = ap.parse_args()

    lookup = {}
    dup = 0
    short_only = 0
    for path in args.inputs:
        data = json.load(open(path))
        for r in data:
            key = r["path"]
            caps = r.get("cap") or []
            short = caps[0] if len(caps) >= 1 else None
            long = caps[1] if len(caps) >= 2 else short  # degrade gracefully
            if len(caps) < 2:
                short_only += 1
            if key in lookup:
                dup += 1
            lookup[key] = {"short": short, "long": long}
        print(f"  {path}: {len(data)} records")

    with open(args.out, "w") as f:
        json.dump(lookup, f, ensure_ascii=False)
    print(f"wrote {len(lookup)} keys -> {args.out}  (duplicates merged: {dup}, single-caption: {short_only})")
    # sample
    k = next(iter(lookup))
    print(f"sample key: {k}")
    print(f"  short: {lookup[k]['short'][:80]!r}")
    print(f"  long : {lookup[k]['long'][:80]!r}")


if __name__ == "__main__":
    main()

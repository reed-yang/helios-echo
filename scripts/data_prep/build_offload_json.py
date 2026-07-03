#!/usr/bin/env python3
"""Build a Helios offload-format JSON from the streaming_model_data corpus.

Joins train.jsonl (clip ids) with manifest.csv (resolution/duration) and emits the
toy_filter-format records that tools/offload_data/get_short-latents.py consumes
(cut/crop/fps/num_frames/resolution/cap/path). Source videos are the 24fps re-encodes.

Frame count is computed at the TARGET fps (24) as round(duration*24); this exactly
matches the actual frame count of the videos_24fps re-encodes (verified). Clips with
fewer than --min-frames frames are dropped, because the Stage-1 history-latents
dataloader hard-drops num_frame < 121.

The `cap` field is set to one chosen caption version as a compat/fallback; the actual
4-version prompt mixing is done by the (patched) encoder, which reads
prompts_rewritten.json directly keyed by clip id.

Example:
  python scripts/data_prep/build_offload_json.py \
    --root /mnt/beegfs/dataset/streaming_model_data \
    --split train.jsonl --videos-subdir videos_24fps \
    --out /mnt/beegfs/dataset/streaming_model_data/offload_train_24fps.json
"""
import argparse
import csv
import json
import os


def load_manifest(path):
    by_file = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_file[r["file"]] = r
    return by_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/beegfs/dataset/streaming_model_data")
    ap.add_argument("--split", default="train.jsonl", help="jsonl with a 'clip' field per line (relative to --root)")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--rewritten", default="prompts_rewritten.json")
    ap.add_argument("--videos-subdir", default="videos_24fps")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--min-frames", type=int, default=121)
    ap.add_argument(
        "--cap-version",
        default="medium",
        choices=["ultra_short", "short", "medium", "long"],
        help="caption version written to the JSON 'cap' field (compat/fallback only)",
    )
    ap.add_argument("--max-clips", type=int, default=0, help="cap output size for a smoke subset (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = args.root
    manifest = load_manifest(os.path.join(root, args.manifest))
    rewritten = json.load(open(os.path.join(root, args.rewritten)))
    videos_dir = os.path.join(root, args.videos_subdir)

    n_total = n_no_meta = n_no_video = n_no_cap = n_short = 0
    records = []
    with open(os.path.join(root, args.split)) as f:
        for line in f:
            n_total += 1
            clip = json.loads(line)["clip"]  # e.g. "abc.mp4"
            r = manifest.get(clip)
            if r is None:
                n_no_meta += 1
                continue
            try:
                num_frames = round(float(r["duration"]) * args.fps)
                width = int(r["width"])
                height = int(r["height"])
            except (KeyError, ValueError):
                n_no_meta += 1
                continue
            if num_frames < args.min_frames:
                n_short += 1
                continue
            if not os.path.exists(os.path.join(videos_dir, clip)):
                n_no_video += 1
                continue
            caps = rewritten.get(clip)
            if not caps or not caps.get(args.cap_version):
                n_no_cap += 1
                continue
            records.append(
                {
                    "cut": [0, num_frames],
                    "crop": [0, width, 0, height],
                    "fps": args.fps,
                    "num_frames": num_frames,
                    "resolution": {"height": height, "width": width},
                    "cap": [caps[args.cap_version]],
                    "path": clip,
                }
            )
            if args.max_clips and len(records) >= args.max_clips:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f)

    print(f"split lines:        {n_total}")
    print(f"dropped no-metadata:{n_no_meta}")
    print(f"dropped <{args.min_frames} frames: {n_short}")
    print(f"dropped no-video:   {n_no_video}")
    print(f"dropped no-caption: {n_no_cap}")
    print(f"RETAINED:           {len(records)} ({100*len(records)/max(n_total,1):.1f}%)")
    print(f"wrote -> {args.out}")
    if records:
        print("sample entry:")
        print(json.dumps(records[0], ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()

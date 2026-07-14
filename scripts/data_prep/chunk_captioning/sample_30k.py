#!/usr/bin/env python3
"""Sample 30k clips for chunk-aligned Gemini captioning.

Source population = the 429,634 pre-encoded 368x640 latents
(/mnt/beegfs/dataset/video_single_24FPS/latents_cfr_int_30b_368x640), i.e. the
corpus the LoRA-368 runs train on. We sample CLIPS (not latents) and re-derive
the chunk grid from the manifest `cut`, NOT from the old .pt files — the old
latents carry an unrecorded random temporal offset (see
docs/CHUNK_ALIGNED_CAPTIONING.md pitfall #11) so the new annotation is anchored
at cut_start and the latents will be re-encoded later.

Per record: chunk k covers RGB frames [cut_start + 33k, cut_start + 33(k+1))
of the CFR video (per-clip constant fps, mostly 25 -- NOT uniformly 24; frame-indexed,
never time-indexed), n_chunks = (cut_end - cut_start) // 33.

Usage:
  python sample_30k.py \
      --listing  /mnt/beegfs/dataset/video_single_24FPS/chunk_captions_gemini_30k/latent_listing.txt \
      --manifest /mnt/beegfs/dataset/video_single_24FPS/train_manifest_combined_v3.jsonl \
      --out      /mnt/beegfs/dataset/video_single_24FPS/chunk_captions_gemini_30k/sample_30k.jsonl \
      --n 30000 --seed 1001
"""
import argparse
import json
import os
import random
import re
from collections import Counter

FRAMES_PER_CHUNK = 33
NAME_RE = re.compile(r"^(?P<clip>.+)_(?P<cs>\d+)-(?P<ce>\d+)_(?P<nf>\d+)_368_640\.pt$")

VIDEO_DIRS = [
    "/mnt/beegfs/dataset/video_single_24FPS/videos_cfr_int",
    "/mnt/beegfs/dataset/video_single_24FPS/videos_cfr_int_bprime",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--min_chunks", type=int, default=3)
    args = ap.parse_args()

    # clip_id -> (video_path, original HTML caption) from the v3 manifest
    manifest = {}
    with open(args.manifest) as f:
        for line in f:
            d = json.loads(line)
            manifest[d["clip_id"]] = (d["video"], d.get("prompt", ""))
    print(f"manifest: {len(manifest)} clips")

    pop, skipped = [], Counter()
    with open(args.listing) as f:
        for line in f:
            m = NAME_RE.match(line.strip())
            if not m:
                skipped["unparsed_name"] += 1
                continue
            clip, cs, ce = m.group("clip"), int(m.group("cs")), int(m.group("ce"))
            n_chunks = (ce - cs) // FRAMES_PER_CHUNK
            if n_chunks < args.min_chunks:
                skipped["too_short"] += 1
                continue
            entry = manifest.get(clip)
            if entry is None:
                skipped["not_in_manifest"] += 1
                continue
            video, hint = entry
            pop.append(
                {
                    "clip_id": clip,
                    "uttid": f"{clip}_{cs}-{ce}",
                    "video": video,
                    "cut": [cs, ce],
                    "n_chunks": n_chunks,
                    "hint": hint,
                }
            )
    print(f"population: {len(pop)}  skipped: {dict(skipped)}")
    assert len(pop) >= args.n, f"population {len(pop)} < requested {args.n}"

    rng = random.Random(args.seed)
    sample = rng.sample(pop, args.n)
    sample.sort(key=lambda d: d["clip_id"])

    # verify video files exist for the sample (cheap: 30k stats)
    missing = [d for d in sample if not os.path.exists(d["video"])]
    if missing:
        print(f"WARNING: {len(missing)} sampled videos missing on disk, e.g. {missing[:3]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for d in sample:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    dist = Counter(d["n_chunks"] for d in sample)
    total_chunks = sum(d["n_chunks"] for d in sample)
    print(f"wrote {len(sample)} -> {args.out}")
    print(f"total chunks: {total_chunks}  (avg {total_chunks/len(sample):.2f}/clip)")
    print("n_chunks distribution (top 15):", dist.most_common(15))


if __name__ == "__main__":
    main()

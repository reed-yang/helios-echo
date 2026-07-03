#!/usr/bin/env python3
"""Build the fixed eval prompt sets for the video_single_24FPS live-eval node.

Input JSONs are offload-format LISTS (train_24fps.json / test_24fps.json): each record
has `path` (bare filename) and `cap = [short, long]`. Per user, eval is FIXED to the
SHORT caption (cap[0]).

- test set: all prompts from test_24fps.json (100).
- train set: a FIXED-RANDOM seeded sample (default 100) from train_24fps.json, chosen
  once and reused every checkpoint so score curves are comparable across steps.

Outputs into --out-dir:
  eval_prompts_test.txt   (one prompt per line; line i -> infer_helios writes i.mp4)
  eval_prompts_train.txt
  eval_meta_test.csv      (id,duration,prompt,clip)  -- id == line index, duration == num_frames
  eval_meta_train.csv
"""
import argparse
import csv
import json
import os
import random


def load_split(path, version_idx):
    rows = []
    data = json.load(open(path))
    for d in data:
        caps = d.get("cap") or []
        if not caps:
            continue
        p = caps[version_idx] if version_idx < len(caps) else caps[0]
        if p:
            rows.append((d["path"], p.replace("\n", " ").strip()))
    return rows


def write_set(rows, out_dir, name, num_frames):
    txt = os.path.join(out_dir, f"eval_prompts_{name}.txt")
    csvp = os.path.join(out_dir, f"eval_meta_{name}.csv")
    with open(txt, "w") as f:
        for _, p in rows:
            f.write(p + "\n")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "duration", "prompt", "clip"])
        for i, (clip, p) in enumerate(rows):
            w.writerow([i, num_frames, p, clip])
    print(f"  {name}: {len(rows)} prompts -> {txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-json", default="/mnt/beegfs/dataset/video_single_24FPS/test_24fps.json")
    ap.add_argument("--train-json", default="/mnt/beegfs/dataset/video_single_24FPS/train_24fps.json")
    ap.add_argument("--version", default="short", choices=["short", "long"], help="cap[0]=short, cap[1]=long")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-frames", type=int, default=330, help="generated video length (multiple of 33)")
    ap.add_argument("--out-dir", default="/mnt/beegfs/xiangbo/helios_runs/eval_prompts_vs24")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    version_idx = 0 if args.version == "short" else 1
    test_all = load_split(args.test_json, version_idx)
    train_all = load_split(args.train_json, version_idx)
    rng = random.Random(args.seed)
    test_rows = test_all if args.n_test >= len(test_all) else rng.sample(test_all, args.n_test)
    train_rows = rng.sample(train_all, min(args.n_train, len(train_all)))

    print(f"caption version: {args.version} (idx {version_idx}); num_frames: {args.num_frames}")
    write_set(test_rows, args.out_dir, "test", args.num_frames)
    write_set(train_rows, args.out_dir, "train", args.num_frames)
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()

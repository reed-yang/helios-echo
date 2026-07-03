#!/usr/bin/env python3
"""Build the fixed eval prompt sets for the live eval node.

- test set: all prompts from test.jsonl (100).
- train set: a FIXED-RANDOM sample (seeded) from train.jsonl (default 100), chosen once
  and reused every checkpoint so score curves are comparable across steps.

Each split file (train.jsonl / test.jsonl) already carries the 4 rewritten caption
versions inline; we pick one version (default 'medium') for comparability.

Outputs into --out-dir:
  eval_prompts_test.txt   (one prompt per line; line i -> infer_helios writes i.mp4)
  eval_prompts_train.txt
  eval_meta_test.csv      (id,duration,prompt,clip)  -- id == line index, duration == num_frames
  eval_meta_train.csv
The CSVs double as the HeliosBench input_csv (columns id,duration,prompt); videos are
later renamed {id}.mp4 -> {id}_{duration}_{duration}.mp4 to match the metric scripts.
"""
import argparse
import csv
import json
import os
import random


def load_split(path, version):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            p = d.get(version) or d.get("medium") or d.get("short") or d.get("ultra_short")
            if p:
                rows.append((d["clip"], p.replace("\n", " ").strip()))
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
    ap.add_argument("--root", default="/mnt/beegfs/dataset/streaming_model_data")
    ap.add_argument("--version", default="medium", choices=["ultra_short", "short", "medium", "long"])
    ap.add_argument("--n-train", type=int, default=15)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-frames", type=int, default=330, help="generated video length (multiple of 33)")
    ap.add_argument("--out-dir", default="/mnt/beegfs/xiangbo/helios_runs/eval_prompts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_all = load_split(os.path.join(args.root, "test.jsonl"), args.version)
    train_all = load_split(os.path.join(args.root, "train.jsonl"), args.version)
    rng = random.Random(args.seed)
    # Fixed-random subsets (seeded) so score curves are comparable across checkpoints.
    test_rows = rng.sample(test_all, min(args.n_test, len(test_all)))
    train_rows = rng.sample(train_all, min(args.n_train, len(train_all)))

    print(f"caption version: {args.version}; num_frames: {args.num_frames}")
    write_set(test_rows, args.out_dir, "test", args.num_frames)
    write_set(train_rows, args.out_dir, "train", args.num_frames)
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()

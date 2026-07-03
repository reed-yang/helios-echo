#!/usr/bin/env python3
"""Build the fixed multi-event (prompt-switching) eval sets for the live eval node.

Reads the held-out test + train multi-event records produced by
scripts/data_prep/build_multievent_json.py (each carries `cap` = list of per-event prompts and
`meta.switch_frame_index`). Picks a FIXED-RANDOM subset (seeded) once and reuses it every
checkpoint so manual comparisons across steps line up.

Per set (test/train) writes into --out-dir:
  eval_me_{set}.json      -> [{id, num_frames, switch_frame_index, caps}]  (for infer_multievent.py)
  eval_me_meta_{set}.csv  -> id,duration,prompt,n_events  (HeliosBench input_csv; prompt = joined caps)

`num_frames`/`duration` is the GENERATED length = (num_frames // 33) * 33, matching what
infer_multievent.py produces and the {id}_{nf}_ori{nf}.mp4 it writes.
"""
import argparse
import csv
import json
import os
import random

FRAMES_PER_CHUNK = 33


def load_records(path):
    with open(path) as f:
        return json.load(f)


def gen_len(num_frames):
    return max(1, num_frames // FRAMES_PER_CHUNK) * FRAMES_PER_CHUNK


def write_set(records, out_dir, name):
    spec_path = os.path.join(out_dir, f"eval_me_{name}.json")
    csv_path = os.path.join(out_dir, f"eval_me_meta_{name}.csv")
    specs = []
    for r in records:
        vid = r["meta"]["video_id"]
        nf = gen_len(int(r["num_frames"]))
        specs.append(
            {
                "id": vid,
                "num_frames": nf,
                "switch_frame_index": [int(s) for s in r["meta"]["switch_frame_index"]],
                "caps": [c.replace("\n", " ").strip() for c in r["cap"]],
            }
        )
    with open(spec_path, "w") as f:
        json.dump(specs, f)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "duration", "prompt", "n_events"])
        for s in specs:
            w.writerow([s["id"], s["num_frames"], " | ".join(s["caps"]), len(s["caps"])])
    print(f"  {name}: {len(specs)} videos -> {spec_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", default="/mnt/beegfs/dataset/streaming_model_data/multievent_ohv_train.json")
    ap.add_argument("--test-json", default="/mnt/beegfs/dataset/streaming_model_data/multievent_ohv_test.json")
    ap.add_argument("--n-train", type=int, default=15)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default="/mnt/beegfs/xiangbo/helios_runs/eval_prompts_multievent")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_all = load_records(args.test_json)
    train_all = load_records(args.train_json)
    rng = random.Random(args.seed)
    test_rows = rng.sample(test_all, min(args.n_test, len(test_all)))
    train_rows = rng.sample(train_all, min(args.n_train, len(train_all)))

    write_set(test_rows, args.out_dir, "test")
    write_set(train_rows, args.out_dir, "train")
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()

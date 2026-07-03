#!/usr/bin/env python3
"""Build a Helios *multi-event* offload-format JSON from the event-shift annotations.

Reads the v7 event-shift segments
  event_switching_v7_release/dataset/event_shift_versions/openai_v2/event_shift_segments.jsonl
(58,967 segments over 27,160 videos; each = a contiguous [start_sec,end_sec] span on the
SAME OpenHumanVid clips as streaming_model_data, with a terse per-segment `event_text`),
groups segments per video, and emits the toy_filter-format records that
tools/offload_data/get_multievent-latents.py consumes — but with two extra fields used
to teach prompt-switching:
  * cap                       : list[str], one event_text per segment (the per-chunk prompts)
  * meta.switch_frame_index   : list[int], RGB-frame boundaries between consecutive events

Times are real seconds and duration-preserving, so they map to the 24fps re-encodes by
frame = round(sec*24) with no rescaling. Source videos are videos_24fps/ (NOT the
annotation's video_path, which points at zips/). num_frames = round(duration*24) from
manifest.csv (matches the actual 24fps frame count, verified in the previous run).

Filters (this run):
  * keep videos with >= 2 surviving (non-empty) events  -> the prompt-switching signal
  * keep num_frames >= --min-frames (121, the Stage-1 dataloader hard floor)
  * video file must exist in videos_24fps/

Boundaries are forced consistent: each segment's frame span is [round(start*24), round(end*24)),
the last segment's end is snapped to num_frames, zero-length events (from rounding collisions)
are dropped, and switch_frame_index is the start frame of every surviving event after the
first -> strictly increasing in (0, num_frames), len(cap) == len(switch)+1.

Outputs two JSON arrays (same record schema):
  * <out-prefix>_train.json  -> encode these to latents + train on them
  * <out-prefix>_test.json   -> held-out (NOT trained); used by the eval prompt builder

Example:
  python scripts/data_prep/build_multievent_json.py \
    --segments /mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/\
event_switching_v7_release/dataset/event_shift_versions/openai_v2/event_shift_segments.jsonl \
    --root /mnt/beegfs/dataset/streaming_model_data --videos-subdir videos_24fps \
    --out-prefix /mnt/beegfs/dataset/streaming_model_data/multievent_ohv
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict


def load_manifest(path):
    """clip_id (32-hex stem) -> row. Falls back to the 'file' stem if clip_id absent."""
    by_id = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = r.get("clip_id")
            if not key:
                fn = r.get("file", "")
                key = os.path.basename(fn).replace(".mp4", "")
            by_id[key] = r
    return by_id


def build_record(video_id, segs, manifest, videos_dir, fps, min_frames):
    """Return (record, reason_dropped). record is None if dropped."""
    r = manifest.get(video_id)
    if r is None:
        return None, "no_meta"
    try:
        num_frames = round(float(r["duration"]) * fps)
        width = int(r["width"])
        height = int(r["height"])
    except (KeyError, ValueError):
        return None, "no_meta"
    if num_frames < min_frames:
        return None, "short"
    clip = f"{video_id}.mp4"
    if not os.path.exists(os.path.join(videos_dir, clip)):
        return None, "no_video"

    segs = sorted(segs, key=lambda s: s["start_sec"])
    # cumulative end frames; snap the final boundary to num_frames
    ends = [round(float(s["end_sec"]) * fps) for s in segs]
    ends[-1] = num_frames
    bounds = [0] + ends

    events = []  # (event_text, lo_frame, hi_frame)
    for i, s in enumerate(segs):
        lo = max(0, min(bounds[i], num_frames))
        hi = max(0, min(bounds[i + 1], num_frames))
        if hi > lo:  # drop zero-length events from rounding collisions
            events.append((s.get("event_text", "") or "", lo, hi))

    if len(events) < 2:
        return None, "lt2_events"
    if any(not e[0].strip() for e in events):
        return None, "empty_text"

    cap = [e[0] for e in events]
    switch = [e[1] for e in events[1:]]  # start frame of each event after the first
    # sanity: strictly increasing, inside (0, num_frames)
    assert all(0 < switch[i] < num_frames for i in range(len(switch))), (video_id, switch, num_frames)
    assert all(switch[i] < switch[i + 1] for i in range(len(switch) - 1)), (video_id, switch)
    assert len(cap) == len(switch) + 1

    rec = {
        "cut": [0, num_frames],
        "crop": [0, width, 0, height],
        "fps": fps,
        "num_frames": num_frames,
        "resolution": {"height": height, "width": width},
        "cap": cap,
        "path": clip,
        "meta": {"switch_frame_index": switch, "video_id": video_id, "n_events": len(cap)},
    }
    return rec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--segments",
        default="/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/"
        "event_switching_v7_release/dataset/event_shift_versions/openai_v2/event_shift_segments.jsonl",
    )
    ap.add_argument("--root", default="/mnt/beegfs/dataset/streaming_model_data")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--videos-subdir", default="videos_24fps")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--min-frames", type=int, default=121)
    ap.add_argument("--n-test", type=int, default=100, help="held-out test videos (excluded from train)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-clips", type=int, default=0, help="cap TRAIN size for a smoke subset (0 = all)")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>_train.json and <prefix>_test.json")
    args = ap.parse_args()

    manifest = load_manifest(os.path.join(args.root, args.manifest))
    videos_dir = os.path.join(args.root, args.videos_subdir)

    # group segments by video
    segs_by_vid = defaultdict(list)
    n_seg = 0
    with open(args.segments) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            segs_by_vid[d["video_id"]].append(d)
            n_seg += 1

    multi = {v: s for v, s in segs_by_vid.items() if len(s) >= 2}

    reasons = defaultdict(int)
    records = []
    for vid, segs in multi.items():
        rec, why = build_record(vid, segs, manifest, videos_dir, args.fps, args.min_frames)
        if rec is None:
            reasons[why] += 1
        else:
            records.append(rec)

    # deterministic split: reserve n_test for held-out eval, rest for training
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_test = min(args.n_test, len(records))
    test = records[:n_test]
    train = records[n_test:]
    if args.max_clips and len(train) > args.max_clips:
        train = train[: args.max_clips]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    train_path = f"{args.out_prefix}_train.json"
    test_path = f"{args.out_prefix}_test.json"
    with open(train_path, "w") as f:
        json.dump(train, f)
    with open(test_path, "w") as f:
        json.dump(test, f)

    print(f"total segments:         {n_seg}")
    print(f"videos total:           {len(segs_by_vid)}")
    print(f"videos with >=2 segs:   {len(multi)}")
    for k in ("no_meta", "short", "no_video", "lt2_events", "empty_text"):
        print(f"  dropped {k:11s}: {reasons.get(k, 0)}")
    print(f"RETAINED (valid):       {len(records)}")
    print(f"  -> test:  {len(test)}  -> {test_path}")
    print(f"  -> train: {len(train)} -> {train_path}")
    if train:
        s = train[0]
        nev = len(s["cap"])
        print(f"sample train entry: id={s['meta']['video_id']} nf={s['num_frames']} "
              f"n_events={nev} switch={s['meta']['switch_frame_index']}")
        print("  caps:")
        for i, c in enumerate(s["cap"]):
            print(f"    [{i}] {c[:90]}")


if __name__ == "__main__":
    main()

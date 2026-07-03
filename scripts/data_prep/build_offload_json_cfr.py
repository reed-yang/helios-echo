#!/usr/bin/env python3
"""Build a Helios offload-format JSON from the captions_30b_32f / cfr_int manifest.

Source: /mnt/beegfs/dataset/video_single_24FPS/qc/captions_30b_32f/train_manifest_cfr_int.jsonl
Each line is {"clip_id", "video": ".../videos_cfr_int/<id>.mp4", "prompt": "<header>...<event>...<role>..."}.
We read `video` and `prompt` VERBATIM from the manifest (no videos_24fps, no path reconstruction).

Unlike build_offload_json.py, the cfr_int clips are MIXED FPS (24/25/30) and short, so frame count
can NOT be derived as round(duration*24). We PROBE the real frame count + resolution per video with
OpenCV (header read; parallelized over a process pool). Clips with fewer than --min-frames frames are
dropped, because the Stage-1 history-latents dataloader hard-drops num_frame < 121.

Emits the toy_filter-format records get_short-latents.py / BucketedFeatureDataset consume
(cut/crop/fps/num_frames/resolution/cap/path). `cap` is the single manifest prompt as a 1-element list;
`path` is the video BASENAME (the encoder joins it against --video_folder = .../videos_cfr_int).

Example:
  python scripts/data_prep/build_offload_json_cfr.py \
    --manifest /mnt/beegfs/dataset/video_single_24FPS/qc/captions_30b_32f/train_manifest_cfr_int.jsonl \
    --min-frames 121 --workers 64 \
    --out /mnt/beegfs/dataset/video_single_24FPS/offload_cfr_int.json
"""
import argparse
import json
import os
from multiprocessing import Pool


def probe(video_path):
    """Return (num_frames, width, height, fps) or None if unreadable. cv2 header read only."""
    try:
        import cv2

        c = cv2.VideoCapture(video_path)
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(c.get(cv2.CAP_PROP_FPS))
        c.release()
        if n <= 0 or w <= 0 or h <= 0:
            return None
        return (n, w, h, fps)
    except Exception:
        return None


def _probe_one(args):
    idx, video, prompt = args
    meta = probe(video)
    return idx, video, prompt, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="/mnt/beegfs/dataset/video_single_24FPS/qc/captions_30b_32f/train_manifest_cfr_int.jsonl",
    )
    ap.add_argument("--min-frames", type=int, default=121)
    ap.add_argument("--workers", type=int, default=64, help="parallel cv2 header-probe workers")
    ap.add_argument("--max-clips", type=int, default=0, help="cap output size for a smoke subset (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load manifest (verbatim video + prompt).
    items = []  # (idx, video, prompt)
    with open(args.manifest) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            v = d.get("video")
            p = d.get("prompt")
            if not v or not p:
                continue
            items.append((i, v, p))
    n_total = len(items)
    print(f"manifest records:   {n_total}  (probing with {args.workers} workers)")

    records = []
    n_unreadable = n_short = 0
    with Pool(args.workers) as pool:
        for done, (idx, video, prompt, meta) in enumerate(pool.imap_unordered(_probe_one, items, chunksize=256)):
            if (done + 1) % 20000 == 0:
                print(f"  probed {done + 1}/{n_total}  kept={len(records)}", flush=True)
            if meta is None:
                n_unreadable += 1
                continue
            num_frames, width, height, fps = meta
            if num_frames < args.min_frames:
                n_short += 1
                continue
            records.append(
                {
                    "cut": [0, num_frames],
                    "crop": [0, width, 0, height],
                    "fps": round(fps, 3) if fps and fps > 0 else 24.0,
                    "num_frames": num_frames,
                    "resolution": {"height": height, "width": width},
                    "cap": [prompt],
                    "path": os.path.basename(video),
                }
            )
            if args.max_clips and len(records) >= args.max_clips:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f)

    print(f"manifest records:   {n_total}")
    print(f"dropped unreadable: {n_unreadable}")
    print(f"dropped <{args.min_frames} frames: {n_short}")
    print(f"RETAINED:           {len(records)} ({100*len(records)/max(n_total,1):.1f}%)")
    print(f"wrote -> {args.out}")
    if records:
        s = dict(records[0])
        s["cap"] = [s["cap"][0][:160] + "..."]
        print("sample entry:")
        print(json.dumps(s, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()

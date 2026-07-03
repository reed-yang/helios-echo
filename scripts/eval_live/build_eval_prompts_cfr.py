#!/usr/bin/env python3
"""Pick N representative prompts from the cfr_int / captions_30b_32f training manifest for live eval.

The 30b captions are multi-line (<header>\\n<event>\\n<role>...); infer_helios --prompt_txt_path is
line-based, so we JOIN each caption to a single line (newlines -> spaces). Prompts are sampled
deterministically (evenly spaced across the manifest) so the SAME prompts are used every checkpoint
-> you can watch the same scenes improve over training. Writes:
  <out-dir>/prompts.txt   (N lines, one joined caption each)
  <out-dir>/meta.tsv      (clip_id <TAB> prompt[:120] per line, for reference)
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/mnt/beegfs/dataset/video_single_24FPS/qc/captions_30b_32f/train_manifest_cfr_int.jsonl")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out-dir", default="/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr")
    args = ap.parse_args()

    lines = [l for l in open(args.manifest) if l.strip()]
    total = len(lines)
    # Evenly spaced, deterministic indices across the corpus for variety.
    idxs = [int(round(i * (total - 1) / (args.n - 1))) for i in range(args.n)] if args.n > 1 else [0]
    os.makedirs(args.out_dir, exist_ok=True)
    ptxt = os.path.join(args.out_dir, "prompts.txt")
    meta = os.path.join(args.out_dir, "meta.tsv")
    with open(ptxt, "w") as fp, open(meta, "w") as fm:
        for i in idxs:
            d = json.loads(lines[i])
            cid = d.get("clip_id", "")
            prompt = " ".join(str(d["prompt"]).split())  # collapse all whitespace/newlines -> single line
            fp.write(prompt + "\n")
            fm.write(f"{cid}\t{prompt[:120]}\n")
    print(f"manifest lines: {total}; sampled {len(idxs)} prompts (indices {idxs[:5]}...{idxs[-2:]})")
    print(f"wrote {ptxt} and {meta}")
    print("--- first 2 prompts ---")
    for i in idxs[:2]:
        d = json.loads(lines[i]); print(" ".join(str(d["prompt"]).split())[:200])


if __name__ == "__main__":
    main()

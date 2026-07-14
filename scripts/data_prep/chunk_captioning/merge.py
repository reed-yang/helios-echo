#!/usr/bin/env python3
"""Merge annotation shards -> captions_chunks.jsonl (dedupe by clip_id, OK-first)
+ missing.txt (clips still to run, feed back via --sample resume)."""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--shards", required=True, help="dir with part-*.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wanted = {json.loads(l)["clip_id"] for l in open(args.sample)}
    best = {}
    for p in sorted(Path(args.shards).glob("part-*.jsonl")):
        with p.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                cid = d.get("clip_id")
                if cid not in wanted:
                    continue
                ok = bool(d.get("captions")) and not d.get("error")
                if cid not in best or (ok and not (best[cid].get("captions") and not best[cid].get("error"))):
                    best[cid] = d

    ok_records = [d for d in best.values() if d.get("captions") and not d.get("error")]
    ok_records.sort(key=lambda d: d["clip_id"])
    with open(args.out, "w") as f:
        for d in ok_records:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    missing = sorted(wanted - {d["clip_id"] for d in ok_records})
    miss_path = Path(args.out).with_name("missing.txt")
    miss_path.write_text("\n".join(missing) + ("\n" if missing else ""))

    errs = Counter(d["error"].split(":")[0] for d in best.values() if d.get("error"))
    flags = Counter(fl for d in ok_records for fl in d.get("flags", []))
    n_caps = sum(len(d["captions"]) for d in ok_records)
    print(f"ok {len(ok_records)}/{len(wanted)}  chunk captions {n_caps}  missing {len(missing)} -> {miss_path}")
    print(f"error types: {dict(errs)}")
    print(f"lint flags:  {dict(flags)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

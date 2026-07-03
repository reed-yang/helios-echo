#!/usr/bin/env python3
"""Partition the REMAINING 368 encode work into disjoint per-node shards.

Why: one big 56-rank encode job starves on beegfs reads (GPUs at 0%); a single-node 8-rank job runs at
~100%. So we re-cut the not-yet-done clips into one shard per node, each its own single-GPU-per-rank job
reading a DISJOINT set of files. Reuses the already-built full caches (no 1.2M-file os.walk) by slicing
samples+buckets, and reconstructs a matching offload-json subset per shard.
"""
import collections
import json
import os
import pickle

D = "/mnt/beegfs/dataset/video_single_24FPS"
OUT = f"{D}/latents_cfr_int_30b_368x640"
SHARD_DIR = f"{D}/shards_368"
os.makedirs(SHARD_DIR, exist_ok=True)
NODES = ["mc-node01", "mc-node02", "c-node03", "c-node04", "c-node05", "c-node06", "c-node07", "c-node08"]

# 1) done uttids from existing .pt (name = {uttid}_{nf}_{h}_{w}.pt; uttid = all but last 3 fields)
done = set()
for f in os.listdir(OUT):
    if f.endswith(".pt"):
        done.add("_".join(f[:-3].split("_")[:-3]))
print(f"done .pt on disk: {len(done)}")

sources = [
    ("cfrint", f"{D}/offload_cfrint_368_cache.pkl", f"{D}/offload_cfrint_368.json", f"{D}/videos_cfr_int"),
    ("bprime", f"{D}/offload_bprime_368_cache.pkl", f"{D}/offload_bprime_368.json", f"{D}/videos_cfr_int_bprime"),
]

loaded = {}
for tag, cachep, jsonp, folder in sources:
    cache = pickle.load(open(cachep, "rb"))
    samples = cache["samples"]
    data = json.load(open(jsonp))
    rec = {}
    for r in data:
        base = os.path.basename(r["path"]).replace(".mp4", "")
        rec[f"{base}_{r['cut'][0]}-{r['cut'][1]}"] = r
    remaining = [s for s in samples if s["uttid"] not in done]
    loaded[tag] = dict(remaining=remaining, rec=rec, folder=folder)
    print(f"{tag}: total={len(samples)} remaining={len(remaining)}")

rc = len(loaded["cfrint"]["remaining"])
rb = len(loaded["bprime"]["remaining"])
tot = rc + rb
# allocate 8 nodes proportional to remaining; guarantee >=1 to any non-empty source
nc = 0 if rc == 0 else max(1, min(len(NODES) - (1 if rb else 0), round(len(NODES) * rc / tot)))
nb = len(NODES) - nc
if rb == 0:
    nc, nb = len(NODES), 0
print(f"node allocation: cfrint={nc} bprime={nb}")

def make_shards(tag, n, node_slice):
    rem = loaded[tag]["remaining"]; rec = loaded[tag]["rec"]; folder = loaded[tag]["folder"]
    out = []
    for i in range(n):
        sub = rem[i::n]  # round-robin -> even sizes + mixed clip lengths per shard
        buckets = collections.defaultdict(list)
        for j, s in enumerate(sub):
            buckets[s["bucket_key"]].append(j)
        jpath = f"{SHARD_DIR}/{tag}_sh{i}.json"
        cpath = f"{SHARD_DIR}/{tag}_sh{i}_cache.pkl"
        json.dump([rec[s["uttid"]] for s in sub], open(jpath, "w"))
        pickle.dump({"samples": sub, "buckets": dict(buckets)}, open(cpath, "wb"))
        out.append((node_slice[i], jpath, folder, len(sub)))
    return out

shards = make_shards("cfrint", nc, NODES[:nc]) + make_shards("bprime", nb, NODES[nc:nc + nb])
with open(f"{SHARD_DIR}/shardmap.tsv", "w") as f:
    for node, jpath, folder, n in shards:
        f.write(f"{node}\t{jpath}\t{folder}\t{n}\n")
        print(f"  {node}  n={n:>6}  {os.path.basename(jpath)}  ({os.path.basename(folder)})")
print(f"\nwrote {len(shards)} shards -> {SHARD_DIR}/shardmap.tsv")

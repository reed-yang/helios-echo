#!/usr/bin/env python
"""Filter an offload manifest down to clips NOT yet encoded, so a (re)launched encode never wastes
GPU/IO re-reading finished videos just to skip them (the idempotent restart's hidden cost).

A latent .pt is named  {uttid}_{num_frame}_{h}_{w}.pt  where uttid = <clip>_<cutstart>-<cutend>.
A manifest clip's uttid = basename(path without .mp4) + f"_{cut[0]}-{cut[1]}".

  python scripts/wan22/filter_undone.py <manifest.json> <latents_out_folder> <undone_manifest.json>
"""
import json
import os
import sys

manifest, out_folder, undone_manifest = sys.argv[1:4]

done = set()
for f in os.listdir(out_folder):
    if f.endswith(".pt"):
        done.add(f[:-3].rsplit("_", 3)[0])  # strip _numframe_h_w.pt -> uttid

data = json.load(open(manifest))
keep = [d for d in data if (os.path.basename(d["path"])[:-4] + f'_{d["cut"][0]}-{d["cut"][1]}') not in done]
json.dump(keep, open(undone_manifest, "w"))
print(f"manifest {len(data)} | already-done .pt {len(done)} | undone -> {len(keep)}  ({undone_manifest})", flush=True)

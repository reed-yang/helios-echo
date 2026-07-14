#!/usr/bin/env python3
"""Build the tagged-caption manifest for the reweight_v2 ("rwtag") run.

For every unique source .pt latent in the 6 selected datasets, record its
caption with a dataset special tag prepended at the very start of the caption
(e.g. "<hatching*>\n<header>..."). The manifest drives
encode_tag_sidecars.py, which re-encodes the tagged captions with UMT5 into
small sidecar files (the .pt latents themselves are NOT copied - beegfs has
no room for a 2.2 TB duplicate; the dataloader overrides embeds at load time
via HELIOS_TAG_EMBED_DIR).

Caption sources:
  realsource / music_performance : <ds>/manifest/captions_all_merged.jsonl (clip_id, caption)
  animal                         : train_filtered/animal/captions.jsonl    (clip_id, caption)
  jesus / genesis / hatching     : read prompt_raw directly from the .pt   (only 2,555 files)
                                   (hatching includes the extension__ subset -> same <hatching*> tag)

Any realsource/music file whose clip_id misses the jsonl join falls back to
reading prompt_raw from the .pt. A random sample of joined entries is
verified against the .pt prompt_raw.

NOTE: ~10% of realsource/music .pt files store an EMPTY prompt_raw (their
prompt_embed is the UMT5 encoding of "", i.e. v1 trained them caption-less)
even though the jsonl has a caption. The jsonl is therefore authoritative:
a .pt/jsonl comparison only counts as a mismatch when the .pt raw is
non-empty and differs.
"""
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor

OUT_DIR = "/mnt/beegfs/xiangbo/helios_data/reweight_v2_meta"
MANIFEST = os.path.join(OUT_DIR, "caption_manifest.jsonl")

# filename = <clip_id>_<start>-<end>_<nframes>_368_640.pt
FN_PAT = re.compile(r"^(?P<clip>.+)_(?P<span>\d+-\d+)_(?P<nf>\d+)_368_640\.pt$")

S2 = "/mnt/beegfs/dataset/live_model_data_s2/organized/split"
TF = "/mnt/beegfs/dataset/live_model_data/train_filtered"
DEMO = "/mnt/beegfs/dataset/demo_data/latents_cfr_int_368x640"

# dataset -> (latent_dir, tag, caption_jsonl or None (=read .pt), filename filter)
SPECS = {
    "realsource": (f"{S2}/realsource/latents", "<realsource*>",
                   f"{S2}/realsource/manifest/captions_all_merged.jsonl", None),
    "music_performance": (f"{S2}/music_performance/latents", "<music_performance*>",
                          f"{S2}/music_performance/manifest/captions_all_merged.jsonl", None),
    "animal": (f"{TF}/animal/latents", "<animal*>", f"{TF}/animal/captions.jsonl", None),
    "jesus": (DEMO, "<jesus*>", None, lambda f: f.startswith("jesus__")),
    "genesis": (DEMO, "<genesis*>", None, lambda f: f.startswith("genesis__")),
    "hatching": (DEMO, "<hatching*>", None,
                 lambda f: f.startswith("hatching__") or f.startswith("extension__")),
}

VERIFY_PER_DS = 20


def read_prompt_raw(path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    raws = d.get("prompt_raws")
    n_events = 0
    if raws is not None:
        n_events = len(raws)
        assert n_events == 1, f"{path}: {n_events} events, expected 1 (single-prompt equivalence)"
    return path, d["prompt_raw"], n_events, "event_prompt_embeds" in d


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    entries = []          # dicts
    fallback = []         # (dataset, tag, path) needing .pt read
    seen_base = {}

    for ds, (latent_dir, tag, cap_jsonl, filt) in SPECS.items():
        files = sorted(f for f in os.listdir(latent_dir)
                       if f.endswith(".pt") and FN_PAT.match(f) and (filt is None or filt(f)))
        caps = {}
        if cap_jsonl:
            with open(cap_jsonl) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get("caption"):
                        caps[r["clip_id"]] = r["caption"]
        n_join = n_fb = 0
        for fn in files:
            base = fn
            if base in seen_base:
                raise SystemExit(f"basename collision: {base} in {ds} and {seen_base[base]}")
            seen_base[base] = ds
            path = os.path.join(latent_dir, fn)
            clip = FN_PAT.match(fn).group("clip")
            cap = caps.get(clip)
            if cap_jsonl and cap:
                entries.append({"base": base, "path": path, "dataset": ds, "tag": tag,
                                "caption": cap, "src": "jsonl"})
                n_join += 1
            else:
                fallback.append((ds, tag, path))
                n_fb += 1
        print(f"{ds:18s} files={len(files):6d} joined={n_join:6d} fallback_to_pt={n_fb:6d}", flush=True)

    # Read prompt_raw from .pt for demo datasets + any join misses (parallel).
    if fallback:
        print(f"reading prompt_raw from {len(fallback)} .pt files ...", flush=True)
        with ProcessPoolExecutor(max_workers=16) as ex:
            for (ds, tag, path), (p2, raw, n_ev, has_ev) in zip(
                    fallback, ex.map(read_prompt_raw, [f[2] for f in fallback], chunksize=8)):
                entries.append({"base": os.path.basename(path), "path": path, "dataset": ds,
                                "tag": tag, "caption": raw, "src": "pt",
                                "has_event_embeds": has_ev})

    # Verify a sample of jsonl-joined captions against the .pt prompt_raw.
    joined = [e for e in entries if e["src"] == "jsonl"]
    random.seed(1001)
    sample = []
    for ds in SPECS:
        pool = [e for e in joined if e["dataset"] == ds]
        sample += random.sample(pool, min(VERIFY_PER_DS, len(pool)))
    bad = empty_pt = 0
    if sample:
        print(f"verifying {len(sample)} jsonl captions against .pt prompt_raw ...", flush=True)
        with ProcessPoolExecutor(max_workers=16) as ex:
            for e, (_, raw, _, _) in zip(sample, ex.map(read_prompt_raw,
                                                        [e["path"] for e in sample], chunksize=2)):
                if not raw.strip():
                    empty_pt += 1  # v1 trained this caption-less; jsonl caption fixes it
                elif e["caption"].strip() != raw.strip():
                    bad += 1
                    print(f"MISMATCH {e['base']}\n  jsonl: {e['caption'][:120]}\n  pt   : {raw[:120]}")
    if bad:
        raise SystemExit(f"{bad}/{len(sample)} non-empty caption mismatches - aborting")
    print(f"verification OK ({len(sample)} sampled, 0 non-empty mismatches, "
          f"{empty_pt} empty-.pt-raw fixed by jsonl)")

    for e in entries:
        e["caption_tagged"] = f"{e['tag']}\n{e['caption']}"
    entries.sort(key=lambda e: e["base"])
    with open(MANIFEST, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    from collections import Counter
    n_empty = sum(1 for e in entries if not e["caption"].strip())
    print(f"wrote {len(entries)} entries -> {MANIFEST} (empty captions: {n_empty} -> tag-only)")
    print(dict(Counter(e["dataset"] for e in entries)))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Materialize the reweight_v2 ("rwtag") data mix as symlink folders.

v2 mix (user-specified 2026-07-06): hatching(+extension) x200, genesis x50,
jesus x30, animal x10, music_performance x2, realsource x1. Nothing else
(spatialVID / sekai / ikea_asm / video_single are EXCLUDED).

Same mechanism as v1 (scratchpad build_reweight_tree.py): every source .pt is
symlinked `mult` times; replica k>=1 inserts `repK` BEFORE the last 3
underscore fields so the dataloader's filename parse (nframes, h, w =
parts[-3:]) and the (nf,h,w) bucket are unchanged.

Captions are NOT baked in here: the run reads tagged UMT5 sidecars via
HELIOS_TAG_EMBED_DIR (see encode_tag_sidecars.py) keyed by the resolved
source basename, which is invariant across replicas.
"""
import os
import re
from concurrent.futures import ProcessPoolExecutor

PAT = re.compile(r'^(?P<stem>.+)_(?P<nf>\d+)_368_640\.pt$')
DEST = '/mnt/beegfs/xiangbo/helios_data/reweight_v2'

S2 = '/mnt/beegfs/dataset/live_model_data_s2/organized/split'
TF = '/mnt/beegfs/dataset/live_model_data/train_filtered'
DEMO = '/mnt/beegfs/dataset/demo_data/latents_cfr_int_368x640'

# name -> (source_dir, multiplier, filename prefix filter tuple or None)
SPECS = {
    'realsource':        (f'{S2}/realsource/latents', 1, None),
    'music_performance': (f'{S2}/music_performance/latents', 2, None),
    'animal':            (f'{TF}/animal/latents', 10, None),
    'demo_jesus':        (DEMO, 30, ('jesus__',)),
    'demo_genesis':      (DEMO, 50, ('genesis__',)),
    'demo_hatching':     (DEMO, 200, ('hatching__', 'extension__')),
}


def build_one(item):
    name, (src, mult, filt) = item
    dst = os.path.join(DEST, name)
    os.makedirs(dst, exist_ok=True)
    files = [f for f in os.listdir(src) if f.endswith('.pt') and PAT.match(f) and (filt is None or f.startswith(filt))]
    made = skipped = 0
    for fn in files:
        m = PAT.match(fn)
        stem, tail = m.group('stem'), fn[len(m.group('stem')):]  # tail = _<nf>_368_640.pt
        target = os.path.join(src, fn)
        for k in range(mult):
            link_name = fn if k == 0 else f'{stem}_rep{k}{tail}'
            link = os.path.join(dst, link_name)
            try:
                os.symlink(target, link)
                made += 1
            except FileExistsError:
                skipped += 1
    return name, len(files), mult, made, skipped


if __name__ == '__main__':
    os.makedirs(DEST, exist_ok=True)
    total = 0
    with ProcessPoolExecutor(max_workers=len(SPECS)) as ex:
        for name, nsrc, mult, made, skipped in ex.map(build_one, SPECS.items()):
            total += made + skipped
            print(f'{name:18s} src={nsrc:7d} x{mult:<3d} -> links={made + skipped:7d} (new={made}, existed={skipped})', flush=True)
    print(f'TOTAL links: {total}')

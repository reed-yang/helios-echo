#!/usr/bin/env python3
"""Re-encode tagged captions into UMT5 embedding SIDECAR files (reweight_v2).

Reads caption_manifest.jsonl (build_tag_caption_manifest.py) and, for every
unique source .pt latent, encodes "<dataset*>\n<original caption>" with the
same encode_prompt() used by tools/offload_data/get_text-embedding.py
(prompt_clean -> UMT5, max_len 512, padding zeroed, bf16). The result is
saved TRIMMED to its true sequence length (~2 MB instead of 4.2 MB; the
dataloader re-pads with zeros, which is exactly what encode_prompt produced).

Sidecars are small separate files because beegfs cannot hold a 2.2 TB copy
of the latents; dataloader_history_latents_dist.py overrides the embeds at
load time when HELIOS_TAG_EMBED_DIR points at --out_dir (keyed by resolved
source basename, so all reweight_v2 symlink replicas share one sidecar).

  encode:  python encode_tag_sidecars.py --shard K --num_shards 8
  verify:  python encode_tag_sidecars.py --verify 6
           (per dataset: sidecar exists; untagged re-encode matches the
            embed stored inside the source .pt; tagged embed differs)
"""
import argparse
import json
import os
import random

import torch  # noqa: F401  (import order: torch before anything cv/decord-ish)

MANIFEST = "/mnt/beegfs/xiangbo/helios_data/reweight_v2_meta/caption_manifest.jsonl"
OUT_DIR = "/mnt/beegfs/xiangbo/helios_data/reweight_v2_tagembed"
MODEL_DIR = "/mnt/beegfs/xiangbo/helios_runs/eval_base_init"
MAX_LEN = 512


def load_encoder(device):
    from transformers import AutoTokenizer, UMT5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_DIR, subfolder="text_encoder", dtype=torch.bfloat16)
    text_encoder.eval().requires_grad_(False)
    return tokenizer, text_encoder.to(device)


def encode_batch(tokenizer, text_encoder, prompts, device):
    from helios.utils.utils_base import encode_prompt
    with torch.no_grad():
        embeds, mask = encode_prompt(
            tokenizer=tokenizer, text_encoder=text_encoder, prompt=prompts,
            max_sequence_length=MAX_LEN, device=device, dtype=torch.bfloat16)
    seq_lens = mask.gt(0).sum(dim=1).long().tolist()
    return embeds, seq_lens


def run_encode(args):
    device = "cuda"
    entries = [json.loads(l) for l in open(MANIFEST)]
    entries = [e for i, e in enumerate(entries) if i % args.num_shards == args.shard]
    todo = [e for e in entries if not os.path.exists(os.path.join(OUT_DIR, e["base"]))]
    print(f"shard {args.shard}/{args.num_shards}: {len(entries)} entries, {len(todo)} to encode", flush=True)
    if not todo:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer, text_encoder = load_encoder(device)
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        embeds, seq_lens = encode_batch(tokenizer, text_encoder,
                                        [e["caption_tagged"] for e in chunk], device)
        for e, emb, L in zip(chunk, embeds, seq_lens):
            out = os.path.join(OUT_DIR, e["base"])
            tmp = out + f".tmp{args.shard}"
            torch.save({"base": e["base"], "dataset": e["dataset"], "tag": e["tag"],
                        "prompt_raw_tagged": e["caption_tagged"], "seq_len": int(L),
                        "prompt_embed": emb[:max(int(L), 1)].cpu().clone()},
                       tmp, pickle_protocol=4)
            os.replace(tmp, out)
        if (i // args.batch_size) % 20 == 0:
            print(f"shard {args.shard}: {i + len(chunk)}/{len(todo)}", flush=True)
    print(f"shard {args.shard}: DONE {len(todo)}", flush=True)


def run_verify(args):
    device = "cuda"
    entries = [json.loads(l) for l in open(MANIFEST)]
    by_ds = {}
    for e in entries:
        by_ds.setdefault(e["dataset"], []).append(e)
    tokenizer, text_encoder = load_encoder(device)
    random.seed(7)
    n_bad = 0
    for ds, pool in sorted(by_ds.items()):
        for e in random.sample(pool, min(args.verify, len(pool))):
            sc_path = os.path.join(OUT_DIR, e["base"])
            assert os.path.exists(sc_path), f"missing sidecar {sc_path}"
            sc = torch.load(sc_path, map_location="cpu", weights_only=False)
            src = torch.load(e["path"], map_location="cpu", weights_only=False)
            ok = []
            # 1) our encoder reproduces the ORIGINAL stored embed from the untagged caption
            #    (only meaningful when the .pt was encoded with a non-empty caption)
            if src["prompt_raw"].strip() and src["prompt_raw"].strip() == e["caption"].strip():
                ref = src["prompt_embed"].float()
                re_emb, seq_lens = encode_batch(tokenizer, text_encoder, [e["caption"]], device)
                re_emb = re_emb[0].float().cpu()
                L = seq_lens[0]
                cos = torch.nn.functional.cosine_similarity(re_emb[:L], ref[:L], dim=-1).min().item()
                ok.append(f"untag_cos={cos:.4f}")
                if cos < 0.995:
                    n_bad += 1
                    ok.append("**BAD**")
            # 2) the tagged sidecar embed must DIFFER from the original
            L2 = min(sc["seq_len"], src["prompt_embed"].shape[0])
            diff = (sc["prompt_embed"][:L2].float() - src["prompt_embed"][:L2].float()).abs().mean().item()
            ok.append(f"tagged_len={sc['seq_len']} tag_diff={diff:.4f}")
            if diff < 1e-4:
                n_bad += 1
                ok.append("**TAG-NOOP**")
            print(f"{ds:18s} {e['base'][:60]:60s} {' '.join(ok)}", flush=True)
    print(f"verify done, bad={n_bad}")
    raise SystemExit(1 if n_bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--verify", type=int, default=0, help="verify N samples per dataset instead of encoding")
    args = ap.parse_args()
    if args.verify:
        run_verify(args)
    else:
        run_encode(args)

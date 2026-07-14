"""Deterministic chunk-aligned latent encoder (chunk-captioning run).

Unlike tools/offload_data/get_multievent-latents.py (which goes through
BucketedFeatureDataset and inherits its RANDOM temporal crop + non-33-multiple
length buckets — docs/CHUNK_ALIGNED_CAPTIONING.md pitfall #11), this encoder
reads EXACTLY frames [cut_start, cut_start + n_chunks*33) of each clip, so the
saved chunk grid is byte-aligned with the per-chunk Gemini captions in
captions_chunks.jsonl. Every chunk is its own "event":
  event_idx_per_chunk = [0, 1, ..., n_chunks-1]
  event_prompt_embeds = one UMT5 embed per chunk caption
  switch_frame_index  = [33, 66, ...]  (relative to the encoded window)

Saved .pt schema matches get_multievent-latents.py exactly, so
dataloader_history_latents_dist.py trains on it unchanged.
Output name: {clip_id}_{cs}-{ce}_{n33}_368_640.pt   (n33 = frames encoded)

Launch = one srun task per GPU with env rendezvous (see encode_chunk_latents.sbatch);
sharding is rank-strided over the caption records, idempotent (skips existing .pt).
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

os.environ.setdefault("HELIOS_READ_TIMEOUT", "120")

import torchvision.transforms as transforms  # noqa: E402
from helios.dataset.dataloader_mp4_dist import read_cut_crop_and_resize  # noqa: E402
from helios.utils.utils_base import encode_prompt  # noqa: E402
from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: E402

from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.training_utils import free_memory  # noqa: E402

FRAMES_PER_CHUNK = 33
FULL_CROP = (0, 10**9, 0, 10**9)  # slice-clamped -> full frame, same as crop=[0,w,0,h]


def load_records(captions_jsonl, exclude_file=None, min_chunks=4):
    # min_chunks=4: the training dataloader hard-drops num_frame < 121, so
    # 3-chunk (99-frame) clips would never be sampled — don't spend GPU on them.
    excl = set()
    if exclude_file and os.path.exists(exclude_file):
        with open(exclude_file) as f:
            excl = {l.strip() for l in f if l.strip()}
    recs = []
    with open(captions_jsonl) as f:
        for line in f:
            d = json.loads(line)
            if d.get("error") or not d.get("captions"):
                continue
            if d["n_chunks"] < min_chunks:
                continue
            if d["clip_id"] in excl:
                continue
            assert len(d["captions"]) == d["n_chunks"], d["clip_id"]
            recs.append(d)
    recs.sort(key=lambda d: d["clip_id"])
    return recs


def decode_clip(rec, height, width):
    cs, _ = rec["cut"]
    n33 = rec["n_chunks"] * FRAMES_PER_CHUNK
    video = read_cut_crop_and_resize(
        rec["video"], f_prime=n33, h_prime=height, w_prime=width,
        stride=1, start_frame=cs, end_frame=cs + n33, crop=FULL_CROP,
    )  # (T, C, H, W) in [-1, 1]
    return video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--exclude", default=None, help="file of clip_ids to skip (heldout)")
    ap.add_argument("--pretrained_model_name_or_path",
                    default="/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled")
    ap.add_argument("--height", type=int, default=368)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    device = torch.cuda.current_device()
    dtype = torch.bfloat16

    recs = load_records(args.captions, args.exclude)
    if args.limit:
        recs = recs[: args.limit]
    mine = recs[rank::world]
    os.makedirs(args.outdir, exist_ok=True)

    def out_path(rec):
        cs, ce = rec["cut"]
        n33 = rec["n_chunks"] * FRAMES_PER_CHUNK
        return os.path.join(args.outdir,
                            f"{rec['clip_id']}_{cs}-{ce}_{n33}_{args.height}_{args.width}.pt")

    todo = [r for r in mine if not os.path.exists(out_path(r))]
    print(f"[rank {rank}/{world}] {len(mine)} assigned, {len(todo)} to encode", flush=True)
    if not todo:
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
    ).eval().requires_grad_(False).to(device)
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    ).eval().requires_grad_(False).to(device)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)

    pool = ThreadPoolExecutor(max_workers=2)

    def submit(i):
        return pool.submit(decode_clip, todo[i], args.height, args.width) if i < len(todo) else None

    fut = submit(0)
    nxt = submit(1)
    done = err = 0
    for i, rec in enumerate(todo):
        try:
            video = fut.result()
        except Exception as e:
            print(f"[rank {rank}] DECODE FAIL {rec['clip_id']}: {type(e).__name__}: {e}", flush=True)
            err += 1
            fut, nxt = nxt, submit(i + 2)
            continue
        fut, nxt = nxt, submit(i + 2)

        try:
            n_chunks = rec["n_chunks"]
            px = video.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=vae.dtype)  # (1,C,T,H,W)
            chunk_latents = []
            with torch.no_grad():
                for c in range(n_chunks):
                    s = c * FRAMES_PER_CHUNK
                    cur = vae.encode(px[:, :, s: s + FRAMES_PER_CHUNK]).latent_dist.sample()
                    chunk_latents.append((cur - latents_mean) * latents_std)
                vae_latent = torch.stack([cl[0] for cl in chunk_latents], dim=0)  # (n_chunks, 16, lw, h/8, w/8)
                event_embeds, _ = encode_prompt(
                    tokenizer=tokenizer, text_encoder=text_encoder,
                    prompt=rec["captions"], device=device,
                )  # (n_chunks, L, 4096)
            first_img = transforms.ToPILImage()(((video[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8))
            switch = [FRAMES_PER_CHUNK * k for k in range(1, n_chunks)]
            to_save = {
                "vae_latent": vae_latent.detach().cpu(),
                "event_prompt_embeds": event_embeds.detach().cpu(),
                "event_idx_per_chunk": torch.arange(n_chunks, dtype=torch.long),
                "switch_frame_index": torch.tensor(switch, dtype=torch.long),
                "first_frames_image": first_img,
                "prompt_raws": list(rec["captions"]),
                "prompt_embed": event_embeds[0].detach().cpu().clone(),
                "prompt_raw": rec["captions"][0],
            }
            tmp = out_path(rec) + f".tmp{rank}"
            torch.save(to_save, tmp)
            os.replace(tmp, out_path(rec))
            done += 1
        except Exception as e:
            print(f"[rank {rank}] ENCODE FAIL {rec['clip_id']}: {type(e).__name__}: {e}", flush=True)
            err += 1
        if (i + 1) % 20 == 0:
            print(f"[rank {rank}] {i+1}/{len(todo)} done={done} err={err}", flush=True)
            free_memory()

    print(f"[rank {rank}] FINISHED done={done} err={err} of {len(todo)}", flush=True)
    pool.shutdown(wait=False)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

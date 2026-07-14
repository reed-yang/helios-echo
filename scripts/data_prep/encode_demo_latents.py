"""Deterministic single-caption latent encoder for the demo_data corpus.

Same frame math as scripts/data_prep/chunk_captioning/encode_chunk_latents.py
(avoids docs/CHUNK_ALIGNED_CAPTIONING.md pitfall #11: reads EXACTLY frames
[cut_start, cut_start + n_chunks*33), chunk-wise VAE), but each clip carries ONE
whole-clip caption (demo_data/caption_gemini), so:
  event_prompt_embeds = (1, L, 4096)   single UMT5 embed
  event_idx_per_chunk = [0, 0, ..., 0]
  switch_frame_index  = []             (single event, no switching)
.pt schema otherwise identical to get_multievent-latents.py, so
dataloader_history_latents_dist.py loads it unchanged.
Output name: {clip_id}_{cs}-{ce}_{n33}_{H}_{W}.pt

Records jsonl: {"clip_id", "video", "cut": [cs, ce], "n_chunks", "caption"}.
No torch.distributed: sharding is rank-strided via SLURM_PROCID/SLURM_NTASKS
(or --rank/--world), idempotent (atomic tmp+rename, skips existing .pt).
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch

os.environ.setdefault("HELIOS_READ_TIMEOUT", "180")

import torchvision.transforms as transforms  # noqa: E402
from helios.dataset.dataloader_mp4_dist import read_cut_crop_and_resize  # noqa: E402
from helios.utils.utils_base import encode_prompt  # noqa: E402
from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: E402

from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.training_utils import free_memory  # noqa: E402

FRAMES_PER_CHUNK = 33
FULL_CROP = (0, 10**9, 0, 10**9)


def decode_clip(rec, height, width):
    cs, _ = rec["cut"]
    n33 = rec["n_chunks"] * FRAMES_PER_CHUNK
    return read_cut_crop_and_resize(
        rec["video"], f_prime=n33, h_prime=height, w_prime=width,
        stride=1, start_frame=cs, end_frame=cs + n33, crop=FULL_CROP,
    )  # (T, C, H, W) in [-1, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pretrained_model_name_or_path",
                    default="/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled")
    ap.add_argument("--height", type=int, default=368)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--min_chunks", type=int, default=4,
                    help="training dataloader hard-drops num_frame < 121")
    ap.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    ap.add_argument("--world", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.cuda.current_device()
    dtype = torch.bfloat16

    recs = []
    with open(args.records) as f:
        for line in f:
            d = json.loads(line)
            if d["n_chunks"] >= args.min_chunks and d.get("caption"):
                recs.append(d)
    recs.sort(key=lambda d: d["clip_id"])
    if args.limit:
        recs = recs[: args.limit]
    mine = recs[args.rank:: args.world]
    os.makedirs(args.outdir, exist_ok=True)

    def out_path(rec):
        cs, ce = rec["cut"]
        n33 = rec["n_chunks"] * FRAMES_PER_CHUNK
        return os.path.join(args.outdir,
                            f"{rec['clip_id']}_{cs}-{ce}_{n33}_{args.height}_{args.width}.pt")

    todo = [r for r in mine if not os.path.exists(out_path(r))]
    print(f"[rank {args.rank}/{args.world}] {len(mine)} assigned, {len(todo)} to encode", flush=True)
    if not todo:
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
            print(f"[rank {args.rank}] DECODE FAIL {rec['clip_id']}: {type(e).__name__}: {e}", flush=True)
            err += 1
            fut, nxt = nxt, submit(i + 2)
            continue
        fut, nxt = nxt, submit(i + 2)

        try:
            n_chunks = rec["n_chunks"]
            px = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W) on CPU
            chunk_latents = []
            with torch.no_grad():
                for c in range(n_chunks):
                    s = c * FRAMES_PER_CHUNK
                    cur = px[:, :, s: s + FRAMES_PER_CHUNK].to(device=device, dtype=vae.dtype)
                    cur = vae.encode(cur).latent_dist.sample()
                    chunk_latents.append(((cur - latents_mean) * latents_std).cpu())
                vae_latent = torch.stack([cl[0] for cl in chunk_latents], dim=0)  # (n_chunks, 16, lw, h/8, w/8)
                event_embeds, _ = encode_prompt(
                    tokenizer=tokenizer, text_encoder=text_encoder,
                    prompt=[rec["caption"]], device=device,
                )  # (1, L, 4096)
            event_embeds = event_embeds.detach().cpu()
            first_img = transforms.ToPILImage()(((video[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8))
            to_save = {
                "vae_latent": vae_latent.detach(),
                "event_prompt_embeds": event_embeds,
                "event_idx_per_chunk": torch.zeros(n_chunks, dtype=torch.long),
                "switch_frame_index": torch.tensor([], dtype=torch.long),
                "first_frames_image": first_img,
                "prompt_raws": [rec["caption"]],
                "prompt_embed": event_embeds[0].clone(),
                "prompt_raw": rec["caption"],
            }
            tmp = out_path(rec) + f".tmp{args.rank}"
            torch.save(to_save, tmp)
            os.replace(tmp, out_path(rec))
            done += 1
        except Exception as e:
            print(f"[rank {args.rank}] ENCODE FAIL {rec['clip_id']}: {type(e).__name__}: {e}", flush=True)
            err += 1
        if (i + 1) % 10 == 0:
            print(f"[rank {args.rank}] {i+1}/{len(todo)} done={done} err={err}", flush=True)
            free_memory()

    print(f"[rank {args.rank}] FINISHED done={done} err={err} of {len(todo)}", flush=True)
    pool.shutdown(wait=False)


if __name__ == "__main__":
    main()

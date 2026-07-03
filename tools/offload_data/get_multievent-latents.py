"""get_multievent-latents.py

Offline latent encoder for Helios Stage-1 *multi-event* (prompt-switching) training.
Mirrors tools/offload_data/get_short-latents.py (same env-based distributed launcher and
BucketedFeatureDataset path), but for each video it additionally:
  * encodes EVERY caption in the record's `cap` list (one per event) -> event_prompt_embeds
  * assigns each 33-RGB-frame (=9-latent) chunk to an event via majority overlap on the
    RGB-frame ranges defined by `meta.switch_frame_index` -> event_idx_per_chunk

The dataloader (helios/dataset/dataloader_history_latents_dist.py) reads event_idx_per_chunk
+ event_prompt_embeds and returns the prompt of the event that owns the sampled chunk, so the
model learns to switch prompts at chunk boundaries. The transformer/forward/loss are unchanged.

Input JSON = the multi-event offload records from scripts/data_prep/build_multievent_json.py
(toy_filter format + `cap: list[str]` + `meta.switch_frame_index: list[int]`).

Saved .pt schema (per clip):
  vae_latent          : (N_chunks, 16, latent_window_size, H/8, W/8)   same layout as Stage 1
  event_prompt_embeds : (N_event, L, 4096)                            one UMT5 embed per caption
  event_idx_per_chunk : (N_chunks,) long                              chunk -> event lookup
  switch_frame_index  : (N_event-1,) long                             passthrough (RGB frames)
  first_frames_image  : PIL.Image
  prompt_raws         : list[str]                                     all captions
  prompt_embed        : (L, 4096)   = event_prompt_embeds[0]   } backward-compat shims so the
  prompt_raw          : str         = cap[0]                    } single-prompt path still loads

Launch (one srun task per GPU, env-based rendezvous — see scripts/data_prep/encode_full.sbatch):
  python tools/offload_data/get_multievent-latents.py \
    --json_file .../multievent_ohv_train.json \
    --video_folder /mnt/beegfs/dataset/streaming_model_data/videos_24fps \
    --output_latent_folder /mnt/beegfs/dataset/streaming_model_data/latents_multievent_384x640 \
    --pretrained_model_name_or_path /mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled \
    --batch_size 4
Run --self_test first (no GPU/video) to verify the chunk-assignment helper.
"""

import argparse
import json
import os
from typing import List

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from accelerate import Accelerator
from helios.dataset.dataloader_mp4_dist import BucketedFeatureDataset, BucketedSampler, collate_fn
from helios.utils.utils_base import encode_prompt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan
from diffusers.training_utils import free_memory


# ============================== majority-overlap helper ==============================
# Ported verbatim from the reference helios/multievent/pipeline_helios_multievent.py so
# offline assignment and inference assignment are byte-identical.


def assign_chunks_to_events(
    num_chunks: int,
    frames_per_chunk: int,
    switch_frame_index: List[int],
    num_frames: int,
) -> List[int]:
    """Per-chunk event assignment via majority overlap on RGB frames.

    Chunk c spans RGB [c*frames_per_chunk, (c+1)*frames_per_chunk); assign it to the event
    range [switch[i-1], switch[i]) with the largest overlap. Ties -> earlier event.
    """
    if len(switch_frame_index) == 0:
        return [0] * num_chunks
    boundaries = [0] + list(switch_frame_index) + [num_frames]
    event_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    result = []
    for c in range(num_chunks):
        chunk_lo = c * frames_per_chunk
        chunk_hi = chunk_lo + frames_per_chunk
        best_event, best_overlap = 0, -1
        for e, (s, t) in enumerate(event_ranges):
            overlap = max(0, min(chunk_hi, t) - max(chunk_lo, s))
            if overlap > best_overlap:
                best_overlap = overlap
                best_event = e
        result.append(best_event)
    return result


def _self_test():
    assert assign_chunks_to_events(2, 33, [39], 66) == [0, 1], "case A"
    assert assign_chunks_to_events(2, 33, [33], 66) == [0, 1], "case B"
    assert assign_chunks_to_events(4, 33, [27, 60], 132) == [0, 1, 2, 2], "case C"
    assert assign_chunks_to_events(2, 33, [1], 66) == [1, 1], "case D"
    assert assign_chunks_to_events(3, 33, [], 99) == [0, 0, 0], "case E"
    assert assign_chunks_to_events(1, 6, [3], 6) == [0], "case F (tie -> earlier)"
    print("assign_chunks_to_events: 6 self-tests passed.")


def setup_distributed_env():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _build_side_table(json_file):
    """uttid -> {cap_list, switch_frame_index}. uttid matches BucketedFeatureDataset:
    basename(path)[:-4] + f"_{cut_start}-{cut_end}"."""
    with open(json_file) as f:
        records = json.load(f)
    table = {}
    for rec in records:
        cs, ce = rec["cut"]
        uttid = os.path.basename(rec["path"]).replace(".mp4", "") + f"_{cs}-{ce}"
        sw = rec.get("meta", {}).get("switch_frame_index", [])
        sw = [int(s) for s in (sw if isinstance(sw, list) else [sw])]
        table[uttid] = {"cap_list": list(rec["cap"]), "switch_frame_index": sw}
    return table


def main(
    rank,
    batch_size,
    dataloader_num_workers,
    json_file,
    video_folder,
    output_latent_folder,
    pretrained_model_name_or_path,
    resolution=640,
    stride=1,
):
    weight_dtype = torch.bfloat16
    device = rank
    seed = 42

    side_table = _build_side_table(json_file)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    )

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, weight_dtype
    )

    vae.eval().requires_grad_(False)
    text_encoder.eval().requires_grad_(False)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    dataset = BucketedFeatureDataset(
        json_files=json_file,
        video_folders=video_folder,
        stride=stride,
        force_rebuild=False,
        resolution=resolution,
        single_res=True,
        single_height=384,
        single_width=640,
    )
    sampler = BucketedSampler(dataset, batch_size=batch_size, drop_last=False, shuffle=True, seed=seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2 if dataloader_num_workers != 0 else None,
    )

    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)
    print(f"Dataset size: {len(dataset)}, batches: {len(dataloader)}")
    print(f"Process index: {accelerator.process_index}, World size: {accelerator.num_processes}")

    sampler.set_epoch(0)
    pbar = tqdm(total=len(dataloader), desc="multievent prep") if rank == 0 else None

    latent_window_size = 9
    frame_window_size = (latent_window_size - 1) * 4 + 1  # 33

    for idx, batch in enumerate(dataloader):
        if batch is None or batch.get("videos", None) is None or batch.get("uttid", None) is None:
            if pbar is not None:
                pbar.update(1)
            continue
        free_memory()

        # skip clips already saved
        valid = []
        for i, (uttid, num_frame, height, width) in enumerate(
            zip(
                batch["uttid"],
                batch["video_metadata"]["num_frames"],
                batch["video_metadata"]["height"],
                batch["video_metadata"]["width"],
            )
        ):
            os.makedirs(output_latent_folder, exist_ok=True)
            out_path = os.path.join(output_latent_folder, f"{uttid}_{num_frame}_{height}_{width}.pt")
            if not os.path.exists(out_path) and uttid in side_table:
                valid.append(i)
        if not valid:
            if pbar is not None:
                pbar.update(1)
            continue

        sub_videos = torch.stack([batch["videos"][i] for i in valid])
        sub_first = torch.stack([batch["first_frames_images"][i] for i in valid])
        sub_uttid = [batch["uttid"][i] for i in valid]
        sub_nf = [batch["video_metadata"]["num_frames"][i] for i in valid]
        sub_h = [batch["video_metadata"]["height"][i] for i in valid]
        sub_w = [batch["video_metadata"]["width"][i] for i in valid]

        with torch.no_grad():
            pixel_values = sub_videos.permute(0, 2, 1, 3, 4).to(dtype=vae.dtype, device=device)
            num_rgb = pixel_values.shape[2]
            num_chunks = num_rgb // frame_window_size
            chunk_latents = []
            for c in range(num_chunks):
                s = c * frame_window_size
                cur = pixel_values[:, :, s : s + frame_window_size, :, :]
                cur_latent = vae.encode(cur).latent_dist.sample()
                cur_latent = (cur_latent - latents_mean) * latents_std
                chunk_latents.append(cur_latent)
            vae_latents = torch.stack(chunk_latents, dim=1)  # [B, n_chunks, 16, lw, h, w]
            first_imgs = [transforms.ToPILImage()(x.to(torch.uint8)) for x in sub_first]

        for i in range(len(valid)):
            uttid = sub_uttid[i]
            meta = side_table[uttid]
            cap_list = meta["cap_list"]
            switch = list(meta["switch_frame_index"])
            cur_vae_latent = vae_latents[i].detach().cpu()
            n_chunks_i = cur_vae_latent.shape[0]

            with torch.no_grad():
                event_embeds, _ = encode_prompt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=cap_list,
                    device=device,
                )  # [N_event, L, D]
            event_embeds = event_embeds.detach().cpu()

            event_idx_per_chunk = assign_chunks_to_events(
                num_chunks=n_chunks_i,
                frames_per_chunk=frame_window_size,
                switch_frame_index=switch,
                num_frames=n_chunks_i * frame_window_size,
            )

            out_path = os.path.join(
                output_latent_folder, f"{uttid}_{sub_nf[i]}_{sub_h[i]}_{sub_w[i]}.pt"
            )
            to_save = {
                "vae_latent": cur_vae_latent,
                "event_prompt_embeds": event_embeds,
                "event_idx_per_chunk": torch.tensor(event_idx_per_chunk, dtype=torch.long),
                "switch_frame_index": torch.tensor(switch, dtype=torch.long),
                "first_frames_image": first_imgs[i],
                "prompt_raws": cap_list,
                # backward-compat shims for the single-prompt dataloader path
                "prompt_embed": event_embeds[0].clone(),
                "prompt_raw": cap_list[0],
            }
            try:
                torch.save(to_save, out_path)
            except Exception as e:
                print(f"[save error] {out_path}: {e}")
                continue

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"batch": idx})
        free_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-event latent encoder for Helios prompt-switching.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled",
        help="local dir with tokenizer/vae/text_encoder (avoids the 32-rank HF snapshot race)",
    )
    parser.add_argument("--json_file", type=str, default=None, help="multi-event offload JSON (list of records)")
    parser.add_argument("--video_folder", type=str, default=None, help="folder of 24fps source mp4s")
    parser.add_argument("--output_latent_folder", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--self_test", action="store_true", help="run the assign_chunks_to_events self-test and exit")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        raise SystemExit(0)

    assert args.json_file and args.video_folder and args.output_latent_folder, (
        "--json_file/--video_folder/--output_latent_folder are required"
    )

    setup_distributed_env()
    device = torch.cuda.current_device()

    main(
        rank=device,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        json_file=args.json_file,
        video_folder=args.video_folder,
        output_latent_folder=args.output_latent_folder,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        resolution=args.resolution,
    )

    dist.barrier()
    dist.destroy_process_group()

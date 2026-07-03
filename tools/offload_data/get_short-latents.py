import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from accelerate import Accelerator
from helios.dataset.dataloader_mp4_dist import BucketedFeatureDataset, BucketedSampler
from helios.dataset.dataloader_mp4_dist import collate_fn as _orig_collate_fn
from helios.utils.utils_base import encode_prompt
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan
from diffusers.training_utils import free_memory

# gc.collect()+empty_cache() 3x/batch is a big SERIAL stall that idles CPU+GPU (node ~88% idle,
# GPU ~25%). Throttle the heavy cleanup to every HELIOS_ENC_FREE_EVERY batches; per-batch dels +
# the CUDA caching allocator (expandable_segments) keep memory bounded between cleanups.
_ENC_FREE_EVERY = max(1, int(os.environ.get("HELIOS_ENC_FREE_EVERY", "1")))


def _maybe_free_memory(idx):
    if idx % _ENC_FREE_EVERY == 0:
        free_memory()


def collate_fn(batch):
    """Defensive wrapper around the shared collate_fn.

    The tiny large-frame buckets (e.g. 481f: 10 samples, 501f: 3) plus ragged
    cross-process dispatch can occasionally place clips of two different frame
    counts in one batch -> the shared collate's torch.stack raises
    'stack expects each tensor to be equal size'. Keep only the majority-frame
    items so stacking never fails; the dropped oddball (at most a handful of
    giant clips) is simply skipped and retried on a later resubmit.
    """
    from collections import Counter

    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    shapes = [tuple(item["videos"].shape) for item in batch]
    if len(set(shapes)) > 1:
        majority = Counter(shapes).most_common(1)[0][0]
        kept = [item for item in batch if tuple(item["videos"].shape) == majority]
        print(f"[safe_collate] mixed-shape batch {set(shapes)} -> kept {len(kept)}/{len(batch)} at {majority}")
        batch = kept
    return _orig_collate_fn(batch)


def setup_distributed_env():
    # Long timeout: with many ranks the first-run dataset cache build (os.walk over a large video
    # folder) can leave a long silent window before the first collective; default 10-30min watchdog
    # would kill the job. Pre-building the cache avoids this, but keep a generous timeout as insurance.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed_env():
    dist.destroy_process_group()


def main(
    rank,
    world_size,
    global_rank,
    stride,
    batch_size,
    dataloader_num_workers,
    json_file,
    video_folder,
    output_latent_folder,
    pretrained_model_name_or_path,
    resolution=640,
    single_height=384,
    single_width=640,
    rewritten=None,
    caption_versions=("ultra_short", "short", "medium", "long"),
):
    weight_dtype = torch.bfloat16
    device = rank
    seed = 42

    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, weight_dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, weight_dtype
    )

    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    # dist.barrier()
    dataset = BucketedFeatureDataset(
        json_files=json_file,
        video_folders=video_folder,
        stride=stride,
        force_rebuild=False,
        resolution=resolution,
        single_res=True,
        single_height=single_height,
        single_width=single_width,
        assume_videos_exist=os.environ.get("HELIOS_ASSUME_VIDEOS_EXIST") == "1",
    )
    sampler = BucketedSampler(dataset, batch_size=batch_size, drop_last=False, shuffle=True, seed=seed)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=2 if dataloader_num_workers != 0 else None,
        # persistent_workers=True if dataloader_num_workers > 0 else False,
    )

    print(len(dataset), len(dataloader))
    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")
    print(f"Process index: {accelerator.process_index}, World size: {accelerator.num_processes}")

    sampler.set_epoch(0)
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Processing")
    # dist.barrier()
    for idx, batch in enumerate(dataloader):
        if batch is None or batch["videos"] is None:
            print("None batch, continuing")
            continue
        _maybe_free_memory(idx)

        valid_indices = []
        valid_uttids = []
        valid_num_frames = []
        valid_heights = []
        valid_widths = []
        valid_videos = []
        valid_prompts = []
        valid_first_frames_images = []

        if batch["uttid"] is None:
            print("None batch, contiuning")
            continue

        for i, (uttid, num_frame, height, width) in enumerate(
            zip(
                batch["uttid"],
                batch["video_metadata"]["num_frames"],
                batch["video_metadata"]["height"],
                batch["video_metadata"]["width"],
            )
        ):
            os.makedirs(output_latent_folder, exist_ok=True)
            output_path = os.path.join(output_latent_folder, f"{uttid}_{num_frame}_{height}_{width}.pt")
            if not os.path.exists(output_path):
                valid_indices.append(i)
                valid_uttids.append(uttid)
                valid_num_frames.append(num_frame)
                valid_heights.append(height)
                valid_widths.append(width)
                valid_videos.append(batch["videos"][i])
                valid_prompts.append(batch["prompts"][i])
                valid_first_frames_images.append(batch["first_frames_images"][i])
            else:
                print(f"skipping {uttid}")

        if not valid_indices:
            print("skipping entire batch!")
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({"batch": idx})
            continue

        batch = None
        del batch
        _maybe_free_memory(idx)

        batch = {
            "uttid": valid_uttids,
            "video_metadata": {"num_frames": valid_num_frames, "height": valid_heights, "width": valid_widths},
            "videos": torch.stack(valid_videos),
            "prompts": valid_prompts,
            "first_frames_images": torch.stack(valid_first_frames_images),
        }

        if len(batch["uttid"]) == 0:
            print("All samples in this batch are already processed, skipping!")
            continue

        with torch.no_grad():
            # Get Vae feature
            pixel_values = batch["videos"].permute(0, 2, 1, 3, 4).to(dtype=vae.dtype, device=device)

            latent_window_size = 9
            frame_window_size = (latent_window_size - 1) * 4 + 1
            num_latent_frames = pixel_values.shape[2]
            num_chunk_to_encode = num_latent_frames // frame_window_size

            history_latent_list = []
            for i in range(num_chunk_to_encode):
                start_idx = i * frame_window_size
                end_idx = start_idx + frame_window_size
                cur_pixel_values = pixel_values[:, :, start_idx:end_idx, :, :]
                with torch.no_grad():
                    cur_latent = vae.encode(cur_pixel_values).latent_dist.sample()
                    cur_latent = (cur_latent - latents_mean) * latents_std
                history_latent_list.append(cur_latent)
            vae_latents = torch.stack(history_latent_list, dim=1)

            # Encode prompts: store ALL caption versions per clip (cheap text-encode),
            # so the training dataloader can mix them. Look up the 4 rewritten versions
            # by clip id derived from uttid (uttid == "<clip>_<cutstart>-<cutend>").
            prompts = batch["prompts"]  # the JSON 'cap' (used as fallback + prompt_raw)

            def _clip_key(uttid):
                return uttid.rsplit("_", 1)[0] + ".mp4"

            per_version_prompts = {v: [] for v in caption_versions}
            for uttid, fallback in zip(batch["uttid"], prompts):
                caps = rewritten.get(_clip_key(uttid)) if rewritten is not None else None
                for v in caption_versions:
                    txt = (caps.get(v) if caps else None) or fallback
                    per_version_prompts[v].append(txt)

            version_embeds = {}
            for v in caption_versions:
                emb, _ = encode_prompt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=per_version_prompts[v],
                    device=device,
                )
                version_embeds[v] = emb

            image_tensor = batch["first_frames_images"]
            images = [transforms.ToPILImage()(x.to(torch.uint8)) for x in image_tensor]

        for i, (
            uttid,
            num_frame,
            height,
            width,
            cur_vae_latent,
            cur_first_frames_image,
            cur_prompt,
        ) in enumerate(
            zip(
                batch["uttid"],
                batch["video_metadata"]["num_frames"],
                batch["video_metadata"]["height"],
                batch["video_metadata"]["width"],
                vae_latents,
                images,
                prompts,
            )
        ):
            output_path = os.path.join(output_latent_folder, f"{uttid}_{num_frame}_{height}_{width}.pt")
            temp_to_save = {
                "vae_latent": cur_vae_latent.cpu().detach(),
                "first_frames_image": cur_first_frames_image,
                "prompt_raw": cur_prompt,
            }
            for v in caption_versions:
                temp_to_save[f"prompt_embed_{v}"] = version_embeds[v][i].cpu().detach()
            # legacy single key for backward compat (prefer medium, else first version)
            temp_to_save["prompt_embed"] = temp_to_save.get(
                "prompt_embed_medium", temp_to_save[f"prompt_embed_{caption_versions[0]}"]
            )
            try:
                torch.save(temp_to_save, output_path)
            except Exception:
                continue
            print(f"save latent to: {output_path}")

        if rank == 0:
            pbar.update(1)
            pbar.set_postfix({"batch": idx})

        pixel_values = None
        prompts = None
        image_tensor = None
        images = None
        vae_latents = None
        vae_latents_2 = None
        image_embeds = None
        version_embeds = None
        batch = None
        valid_indices = None
        valid_uttids = None
        valid_num_frames = None
        valid_heights = None
        valid_widths = None
        valid_videos = None
        valid_prompts = None
        valid_first_frames_images = None
        temp_to_save = None

        del pixel_values
        del prompts
        del image_tensor
        del images
        del vae_latents
        del vae_latents_2
        del image_embeds
        del version_embeds
        del batch
        del valid_indices
        del valid_uttids
        del valid_num_frames
        del valid_heights
        del valid_widths
        del valid_videos
        del valid_prompts
        del valid_first_frames_images
        del temp_to_save

        _maybe_free_memory(idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running model training and data processing.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="BestWishYsh/Helios-Base",
        help="Pretrained model path",
    )
    # Override the toy-data defaults to encode an arbitrary dataset.
    parser.add_argument("--json_file", type=str, default=None, help="offload-format JSON (overrides toy default)")
    parser.add_argument("--video_folder", type=str, default=None, help="folder of source mp4s")
    parser.add_argument("--output_latent_folder", type=str, default=None, help="where to write *.pt latents")
    parser.add_argument(
        "--rewritten_json",
        type=str,
        default=None,
        help="prompts_rewritten.json (dict '<clip>.mp4' -> {ultra_short,short,medium,long}); "
        "if set, all versions are encoded and stored per clip for caption mixing",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=640)
    # Encoded latent spatial size (single_res). Defaults preserve the prior 384x640 behavior;
    # pass --single_height 368 to encode at the new 368x640 resolution.
    parser.add_argument("--single_height", type=int, default=384)
    parser.add_argument("--single_width", type=int, default=640)
    parser.add_argument(
        "--caption_versions",
        type=str,
        default="ultra_short,short,medium,long",
        help="comma-separated caption version keys to look up in --rewritten_json and store as "
        "prompt_embed_<v>. For video_single_24FPS (cap=[short,long]) pass 'short,long'.",
    )
    args = parser.parse_args()
    caption_versions = tuple(v.strip() for v in args.caption_versions.split(",") if v.strip())

    setup_distributed_env()

    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()

    rewritten = json.load(open(args.rewritten_json)) if args.rewritten_json else None

    if args.json_file is not None:
        configs = [(args.json_file, args.video_folder, args.output_latent_folder, args.batch_size, args.resolution)]
    else:
        # backward-compatible toy-data default
        configs = [
            ("example/toy_data/toy_filter.json", "example/toy_data", "example/toy_data/latents_short/toy_data", 4, 640)
        ]

    for json_file, video_folder, output_latent_folder, batch_size, cur_resolution in configs:
        main(
            rank=device,
            world_size=world_size,
            global_rank=global_rank,
            stride=1,
            batch_size=batch_size,
            dataloader_num_workers=args.dataloader_num_workers,
            json_file=json_file,
            video_folder=video_folder,
            output_latent_folder=output_latent_folder,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            resolution=cur_resolution,
            single_height=args.single_height,
            single_width=args.single_width,
            rewritten=rewritten,
            caption_versions=caption_versions,
        )

    dist.barrier()
    dist.destroy_process_group()

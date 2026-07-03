"""infer_multievent.py — generate prompt-SWITCHING eval videos for Helios Stage-1.

Reads a multi-event spec JSON (list of {id, num_frames, switch_frame_index, caps}) produced
by scripts/eval_live/build_eval_prompts_multievent.py and, for each video, generates one clip
that HARD-SWITCHES prompt at the annotated event boundaries — the inference mirror of how the
multi-event latents were built for training.

Mechanism (no architecture change): drive the existing HeliosPipeline with
use_interpolate_prompt=True, interpolation_steps=0 (hard switch, no lerp), and
interpolate_time_list = per-event chunk counts derived from switch_frame_index via the SAME
majority-overlap rule used by the offline encoder (tools/offload_data/get_multievent-latents.py)
and the reference HeliosMultiEventPipeline. Events that win 0 chunks are dropped (their prompt
is unused — consistent with training).

Data-parallel across videos via torchrun (each rank takes ids[rank::world_size]). The pipeline
setup block is copied verbatim from infer_helios.py so behaviour matches the released path.
Output is written pre-named for HeliosBench: {id}_{nf}_ori{nf}.mp4.

Example (7 GPUs, skipping c-node08 GPU1):
  CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 torchrun --nproc_per_node=7 infer_multievent.py \
    --spec_json .../eval_me_test.json --lora_path <checkpoint-N> \
    --base_model_path BestWishYsh/Helios-Base --transformer_path BestWishYsh/Helios-Base \
    --output_folder .../step-N/test
"""
import importlib
import os


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import json
from collections import OrderedDict
from typing import List

import torch
import torch.distributed as dist
from tqdm import tqdm


if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
else:
    torch_npu = None

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)

from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video


# majority-overlap rule — byte-identical to the offline encoder / reference pipeline.
def assign_chunks_to_events(num_chunks, frames_per_chunk, switch_frame_index, num_frames):
    if len(switch_frame_index) == 0:
        return [0] * num_chunks
    boundaries = [0] + list(switch_frame_index) + [num_frames]
    event_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    result = []
    for c in range(num_chunks):
        lo = c * frames_per_chunk
        hi = lo + frames_per_chunk
        best_e, best_ov = 0, -1
        for e, (s, t) in enumerate(event_ranges):
            ov = max(0, min(hi, t) - max(lo, s))
            if ov > best_ov:
                best_ov, best_e = ov, e
        result.append(best_e)
    return result


def build_switch_plan(num_frames: int, switch: List[int], caps: List[str], frames_per_chunk: int = 33):
    """-> (prompt_list, interpolate_time_list, nf_gen). Drops 0-chunk events; counts>=1."""
    total_chunks = max(1, num_frames // frames_per_chunk)
    eidx = assign_chunks_to_events(total_chunks, frames_per_chunk, switch, total_chunks * frames_per_chunk)
    counts = OrderedDict()  # first-seen order == temporal order (eidx is non-decreasing)
    for e in eidx:
        counts[e] = counts.get(e, 0) + 1
    surviving = list(counts.keys())
    prompt_list = [caps[e] for e in surviving]
    time_list = [counts[e] for e in surviving]
    return prompt_list, time_list, total_chunks * frames_per_chunk


def parse_args():
    p = argparse.ArgumentParser(description="Multi-event (prompt-switching) eval generation for Helios.")
    p.add_argument("--spec_json", type=str, required=True, help="list of {id,num_frames,switch_frame_index,caps}")
    p.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Base")
    p.add_argument("--transformer_path", type=str, default="BestWishYsh/Helios-Base")
    p.add_argument("--lora_path", type=str, default=None, help="trained adapter dir; omit for raw-Base baseline")
    p.add_argument("--output_folder", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable_compile", action="store_true")
    p.add_argument("--weight_dtype", type=str, default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()
    args.weight_dtype = getattr(torch, args.weight_dtype)
    os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_available() and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        world_size = dist.get_world_size()
        torch.cuda.set_device(device)
    else:
        rank, device, world_size = 0, torch.device("cuda"), 1

    # ---- pipeline setup (mirrors infer_helios.py) ----
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path, subfolder="transformer", torch_dtype=args.weight_dtype
    )
    if not args.enable_compile:
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    if torch.cuda.get_device_capability()[0] >= 9:
        try:
            transformer.set_attention_backend("_flash_3_hub")
        except Exception:
            transformer.set_attention_backend("flash_hub")
    else:
        transformer.set_attention_backend("flash_hub")

    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path, transformer=transformer, vae=vae, scheduler=scheduler, torch_dtype=args.weight_dtype
    )
    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[1.0])
    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
    pipe = pipe.to(device)

    with open(args.spec_json) as f:
        specs = json.load(f)
    specs = specs[rank::world_size]

    for spec in tqdm(specs, desc=f"[rank {rank}] multievent gen"):
        vid = spec["id"]
        caps = spec["caps"]
        switch = [int(s) for s in spec.get("switch_frame_index", [])]
        prompt_list, time_list, nf_gen = build_switch_plan(int(spec["num_frames"]), switch, caps)
        out_path = os.path.join(args.output_folder, f"{vid}_{nf_gen}_ori{nf_gen}.mp4")
        if os.path.exists(out_path):
            continue
        with torch.no_grad():
            try:
                output = pipe(
                    prompt=prompt_list,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=nf_gen,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(args.seed),
                    history_sizes=[16, 2, 1],
                    num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                    keep_first_frame=True,
                    is_enable_stage2=False,
                    # hard prompt switch at chunk boundaries (no lerp), faithful per-event counts
                    use_interpolate_prompt=True,
                    interpolation_steps=0,
                    interpolate_time_list=time_list,
                ).frames[0]
            except Exception as e:
                print(f"[rank {rank}] FAILED {vid}: {e}")
                continue
        export_to_video(output, out_path, fps=24)
        print(f"[rank {rank}] {vid}: {len(prompt_list)} events, chunks={time_list}, nf={nf_gen} -> {out_path}")

    print(f"[rank {rank}] done. Max mem {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()

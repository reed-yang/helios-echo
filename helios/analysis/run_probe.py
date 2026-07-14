"""
Driver: run Helios inference with the attention probe installed and dump
aggregated per-(chunk, denoise_step, layer) attention statistics to JSON.

Design choices (see docstring in attn_probe.py):
  * Checkpoint = Helios-Base by default. Base is un-distilled (50 steps) so the
    evolution of attention over layers *and* denoise steps is visible; it also
    reflects the model's "natural" attention (no distillation shortcut). The
    distilled model is the deployed variant and can be run with --checkpoint.
  * guidance_scale = 1.0 on purpose: the conditional forward pass computes
    exactly the attention we care about, and CFG only *combines* cond/uncond
    outputs afterwards -- it does not change the per-pass attention. Running at
    1.0 gives a single forward per step (clean chunk/step bookkeeping) whose
    attention is identical to the cond pass at any guidance scale.

Usage:
  CUDA_VISIBLE_DEVICES=2 python -m helios.analysis.run_probe --smoke
  CUDA_VISIBLE_DEVICES=2 python -m helios.analysis.run_probe \
      --out helios/analysis/out/base_full.json --num_frames 231
"""

import argparse
import json
import os

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "no")

import torch

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel

from diffusers.models import AutoencoderKLWan

from helios.analysis import attn_probe as AP


BASE_LOCAL = os.environ.get(
    "HELIOS_BASE_PATH",
    "/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd",
)

PROMPTS = {
    "fish": "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales, its fins moving fluidly as it glides past the coral. A close-up shot with dynamic movement.",
    "train": "A dynamic time-lapse video showing rapidly moving scenery from the window of a speeding train: lush green fields, towering trees, quaint countryside houses and distant mountain ranges rushing past. The camera is static, emphasizing the fast motion outside.",
    "man": "An extreme close-up of a gray-haired man with a beard in his 60s, deep in thought as he sits at a cafe in Paris, his eyes following people offscreen, dressed in a wool coat and beret, cinematic 35mm film, golden hour lighting, shallow depth of field.",
}

NEG = "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality, JPEG compression residue, ugly, deformed, disfigured, still picture, messy background."


def load_pipe(base_path, device, dtype):
    transformer = HeliosTransformer3DModel.from_pretrained(base_path, subfolder="transformer", torch_dtype=dtype)
    # NOTE: we deliberately skip replace_rmsnorm/flash_norm/flash_rope so the
    # eager probe branch uses the exact same rope/norm as the real dispatch.
    cuda_major = torch.cuda.get_device_capability()[0]
    try:
        transformer.set_attention_backend("_flash_3_hub" if cuda_major >= 9 else "flash_hub")
    except Exception:
        try:
            transformer.set_attention_backend("flash_hub")
        except Exception:
            pass
    vae = AutoencoderKLWan.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(base_path, subfolder="scheduler", stages=1)
    pipe = HeliosPipeline.from_pretrained(base_path, transformer=transformer, vae=vae, scheduler=scheduler, torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe, transformer


def build_record_steps(n_steps, k=5):
    if n_steps <= k:
        return set(range(n_steps))
    pts = [round(i * (n_steps - 1) / (k - 1)) for i in range(k)]
    return set(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=BASE_LOCAL)
    ap.add_argument("--ckpt_name", default="Helios-Base")
    ap.add_argument("--out", default="helios/analysis/out/base_probe.json")
    ap.add_argument("--prompts", nargs="+", default=["fish", "train"])
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--num_frames", type=int, default=231)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--n_query_sample", type=int, default=512)
    ap.add_argument("--record_k_steps", type=int, default=5)
    ap.add_argument("--sim_amplify_scale", type=float, default=4.0,
                    help="counterfactual history-key scale when model's is_amplify_history is OFF (1.0 disables)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.num_frames = 99
        args.num_inference_steps = 6
        args.prompts = args.prompts[:1]
        args.record_k_steps = 6
        args.out = args.out.replace(".json", "_smoke.json")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[run_probe] loading {args.ckpt_name} from {args.checkpoint}")
    pipe, transformer = load_pipe(args.checkpoint, device, dtype)
    AP.install(transformer)

    AP.PROBE.n_query_sample = args.n_query_sample
    AP.PROBE.sim_amplify_scale = args.sim_amplify_scale
    AP.PROBE.record_steps = build_record_steps(args.num_inference_steps, args.record_k_steps)
    print(f"[run_probe] recording denoise steps: {sorted(AP.PROBE.record_steps)}")

    all_records = []
    all_full_maps = []
    boundaries_dump = None
    meta_common = {
        "checkpoint": args.ckpt_name,
        "task": "T2V",
        "num_inference_steps": args.num_inference_steps,
        "num_layers": int(len(transformer.blocks)),
        "num_heads": int(transformer.config.num_attention_heads),
        "resolution": f"{args.height}x{args.width}",
        "fps": 24,
        "guidance_scale": args.guidance_scale,
        "n_query_sample": args.n_query_sample,
        "recorded_steps": sorted(AP.PROBE.record_steps),
        "categories": AP.CATEGORIES,
        "category_order_in_sequence": ["long_lowres", "mid", "sink", "short_highres", "noisy"],
        "sim_amplify_scale": args.sim_amplify_scale,
        "notes": ("guidance_scale=1.0 => single (conditional) forward per step; attention equals the cond "
                  "pass at any CFG scale. The released checkpoints ship is_amplify_history=False, so the "
                  "history-amplification mechanism is INACTIVE ('mass' = the model's natural attention). "
                  f"'mass_amplified' is a COUNTERFACTUAL: history keys scaled x{args.sim_amplify_scale} to "
                  "illustrate how the mechanism would redistribute mass (amplify_simulated=true)."),
    }

    for pi, pkey in enumerate(args.prompts):
        prompt = PROMPTS.get(pkey, pkey)
        AP.PROBE.reset_run()
        AP.PROBE.n_query_sample = args.n_query_sample
        AP.PROBE.record_steps = build_record_steps(args.num_inference_steps, args.record_k_steps)
        # choose a few representative full-map cells for the LAST chunk
        n_chunks_est = max(1, (args.num_frames + 32) // 33)
        last_chunk = n_chunks_est - 1
        last_step = max(AP.PROBE.record_steps)
        mid_layer = len(transformer.blocks) // 2
        last_layer = len(transformer.blocks) - 1
        AP.PROBE.full_map_cells = {
            (last_chunk, last_step, 0),
            (last_chunk, last_step, mid_layer),
            (last_chunk, last_step, last_layer),
            (min(1, last_chunk), last_step, mid_layer),
        }

        AP.PROBE.enabled = True
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        print(f"[run_probe] prompt {pi}: '{pkey}' frames={args.num_frames}")
        with torch.no_grad():
            _ = pipe(
                prompt=prompt,
                negative_prompt=NEG,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
                history_sizes=[16, 2, 1],
                num_latent_frames_per_chunk=9,
                keep_first_frame=True,
                is_enable_stage2=False,
            ).frames[0]
        AP.PROBE.enabled = False

        n_chunks = AP.PROBE.cur_chunk + 1
        print(f"[run_probe]   -> {len(AP.PROBE.records)} records over {n_chunks} chunks")
        for r in AP.PROBE.records:
            r["prompt"] = pkey
        all_records.extend(AP.PROBE.records)
        for fm in AP.PROBE.full_maps:
            fm["prompt"] = pkey
        all_full_maps.extend(AP.PROBE.full_maps)
        if boundaries_dump is None:
            boundaries_dump = [
                {"chunk": c, "denoise_step": s, "ranges": rng}
                for (c, s), rng in sorted(AP.PROBE.boundaries_per_step.items())
            ]

    out = {
        "meta": meta_common,
        "boundaries_per_step": boundaries_dump,
        "by_step_layer": all_records,
        "full_maps": all_full_maps,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[run_probe] wrote {len(all_records)} records, {len(all_full_maps)} full maps -> {args.out}")

    # ---- validation summary (smoke) ----
    if all_records:
        rs = [r["row_sum"] for r in all_records]
        print(f"[validate] row_sum mean={sum(rs)/len(rs):.5f} min={min(rs):.5f} max={max(rs):.5f} (want ~1.0)")
        r0 = all_records[0]
        msum = sum(r0["mass"].values())
        print(f"[validate] sample record chunk={r0['chunk']} step={r0['denoise_step']} layer={r0['layer']}")
        print(f"[validate]   mass sum over categories = {msum:.5f} (want ~1.0)")
        print(f"[validate]   mass = { {k: round(v,4) for k,v in r0['mass'].items()} }")
        print(f"[validate]   n_tokens = {r0['n_tokens']}")
        if boundaries_dump:
            b0 = boundaries_dump[0]["ranges"]
            covered = all(b0[c][1] >= b0[c][0] for c in AP.CATEGORIES)
            spans = sorted([tuple(b0[c]) for c in AP.CATEGORIES])
            contiguous = all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1)) and spans[0][0] == 0
            print(f"[validate]   boundaries={b0}")
            print(f"[validate]   contiguous&full-cover from 0 = {contiguous}, all valid = {covered}")


if __name__ == "__main__":
    main()

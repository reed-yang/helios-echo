#!/usr/bin/env python3
"""Consolidate a full-FT DeepSpeed ZeRO-2 checkpoint into a `<ckpt>/transformer/` HF dir for inference.

The full-FT training save hook is kept CLEAN (no in-hook consolidated save — that desynced the collective
save path). So each checkpoint-N holds only the DeepSpeed sharded model+optim under pytorch_model/. This
script gathers them with deepspeed's get_fp32_state_dict_from_zero_checkpoint, loads into a freshly-built
HeliosTransformer3DModel (same transformer_additional_kwargs as the trainer), and save_pretrained ->
checkpoint-N/transformer/ so infer_helios.py --transformer_path checkpoint-N can load it (it appends
subfolder="transformer"). Idempotent: skips if transformer/diffusion_pytorch_model*.safetensors exists.

Usage:
  python tools/offload_data/consolidate_fullft.py <checkpoint-N dir> <base_transformer_init_dir>
"""
import argparse, glob, os, sys

import torch

REPO = "/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team"
sys.path.insert(0, REPO)
from helios.modules.transformer_helios import HeliosTransformer3DModel  # noqa: E402
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint  # noqa: E402

# Must match train_helios.py transformer_additional_kwargs for the cfr full-FT run.
TF_KWARGS = {
    "has_multi_term_memory_patch": True,
    "zero_history_timestep": True,
    "restrict_self_attn": False,
    "guidance_cross_attn": True,
    "is_train_restrict_lora": False,
    "restrict_lora": False,
    "restrict_lora_rank": 128,
    "is_amplify_history": False,
    "history_scale_mode": "per_head",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("base", help="Helios-Base snapshot dir (contains transformer_init/)")
    ap.add_argument("--subfolder", default="transformer_init")
    args = ap.parse_args()

    out = os.path.join(args.ckpt, "transformer")
    if glob.glob(os.path.join(out, "diffusion_pytorch_model*.safetensors")):
        print(f"[consolidate] {out} already exists -> skip")
        return

    print(f"[consolidate] building base transformer from {args.base}/{args.subfolder}")
    model = HeliosTransformer3DModel.from_pretrained(
        args.base, subfolder=args.subfolder, transformer_additional_kwargs=TF_KWARGS, torch_dtype=torch.float32
    )
    print(f"[consolidate] gathering ZeRO fp32 state_dict from {args.ckpt}")
    sd = get_fp32_state_dict_from_zero_checkpoint(args.ckpt)
    # Strip a possible leading "module." (accelerate/deepspeed wrapping) so keys match the bare model.
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[consolidate] loaded: missing={len(missing)} unexpected={len(unexpected)} (sd={len(sd)} params)")
    if missing:
        print("  first missing:", missing[:8])
    if unexpected:
        print("  first unexpected:", unexpected[:8])
    # A correct consolidation should have ~0 missing among the trainable transformer params.
    model.to(torch.bfloat16).save_pretrained(out)
    print(f"[consolidate] wrote {out}")


if __name__ == "__main__":
    main()

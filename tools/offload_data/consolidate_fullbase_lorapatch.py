#!/usr/bin/env python3
"""Consolidate a full_finetune_scope=base_linear_lora_patch DeepSpeed ZeRO-2 checkpoint into an
inference-ready `<ckpt>/transformer/` HF dir.

Unlike consolidate_fullft.py (plain full-FT), this run's training model = merged base + an injected
patch_embedding LoRA adapter + FROZEN norm/AdaLN + FROZEN patch_embedding base. The ZeRO fp32 state
holds only the TRAINABLE params (base-linear, memory patches, patch_embedding.lora_A/B). The frozen
params (patch_embedding.base_layer, norms) are NOT in the optimizer, so we take them from the MERGED
start transformer (which holds their correct, unchanging values). Then we fuse the patch_embedding LoRA
into its base and drop the adapter -> a clean, PEFT-free transformer loadable by infer_helios.py.

Usage:
  python tools/offload_data/consolidate_fullbase_lorapatch.py <checkpoint-N dir> <merged_start_dir>
    [--subfolder transformer]
"""
import argparse, glob, os, sys

import torch
from peft import LoraConfig

REPO = "/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team"
sys.path.insert(0, REPO)
from helios.modules.transformer_helios import HeliosTransformer3DModel  # noqa: E402
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint  # noqa: E402

# Must match train_helios.py transformer_additional_kwargs for this run.
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
LORA_RANK = 128
LORA_ALPHA = 128.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="checkpoint-N dir (contains pytorch_model/ DeepSpeed shards)")
    ap.add_argument("merged", help="merged start dir (contains transformer/ with frozen base+norm values)")
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--out", default="", help="output dir for transformer/ (default <ckpt>/transformer)")
    args = ap.parse_args()

    out = args.out if args.out else os.path.join(args.ckpt, "transformer")
    if glob.glob(os.path.join(out, "diffusion_pytorch_model*.safetensors")):
        print(f"[consolidate] {out} already exists -> skip")
        return

    print(f"[consolidate] building model from merged start {args.merged}/{args.subfolder}")
    model = HeliosTransformer3DModel.from_pretrained(
        args.merged, subfolder=args.subfolder, transformer_additional_kwargs=TF_KWARGS, torch_dtype=torch.float32
    )
    # inject the SAME patch_embedding LoRA structure the trainer used, so the ZeRO state keys match.
    model.add_adapter(
        LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
            init_lora_weights="gaussian", target_modules=["patch_embedding"],
        )
    )

    print(f"[consolidate] gathering ZeRO fp32 state_dict from {args.ckpt}")
    sd = get_fp32_state_dict_from_zero_checkpoint(args.ckpt)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Expected: 'missing' = the frozen params (norms, patch_embedding.base_layer) kept from merged; that's OK.
    #           'unexpected' should be ~0. Trainable base-linear / memory / patch_embedding.lora_* must load.
    print(f"[consolidate] loaded sd={len(sd)} params: missing={len(missing)} unexpected={len(unexpected)}")
    pe_lora_loaded = [k for k in sd if "patch_embedding" in k and "lora_" in k]
    print(f"[consolidate] patch_embedding LoRA keys in sd: {len(pe_lora_loaded)} -> {pe_lora_loaded}")
    if unexpected:
        print("  !! unexpected keys (should be empty):", unexpected[:12])
    # sanity: base-linear must be present in sd (else trained weights not applied)
    n_base_linear = sum(1 for k in sd if k.endswith(".weight") and (".attn" in k or ".ffn" in k))
    print(f"[consolidate] base-linear weight tensors in sd: {n_base_linear} (should be > 0)")
    assert n_base_linear > 0, "no base-linear weights in the ZeRO state -> wrong checkpoint/scope"

    # fuse patch_embedding LoRA into its base Conv3d, drop the adapter -> clean transformer.
    model.fuse_lora(lora_scale=1.0, safe_fusing=True)
    model.unload_lora()

    model.to(torch.bfloat16).save_pretrained(out)
    print(f"[consolidate] wrote {out}")


if __name__ == "__main__":
    main()

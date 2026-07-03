"""Selective LoRA merge for the "full base-linear + LoRA patch_embedding" recipe.

Unlike tools/merge_lora_for_helios.py (which fuses the ENTIRE LoRA adapter into the base), this tool
produces the starting point for a run that full-fine-tunes the base linear backbone while KEEPING
patch_embedding on a LoRA adapter. So it:

  1. bakes every LoRA delta EXCEPT patch_embedding into the base transformer weights (direct tensor
     merge: W += (alpha/r) * lora_B @ lora_A -- all merged modules are Linear),
  2. overwrites patch_short/mid/long with the fully-trained memory patches from transformer_partial.pth,
  3. leaves patch_embedding at its base-init value (NOT merged), and
  4. emits patch_embedding_lora.safetensors (the patch_embedding LoRA keys, verbatim) for the training
     run to seed its live patch_embedding adapter (via model_config.patch_embedding_lora_init_path).

Result: consolidated `<out>/transformer/` (loadable like Helios-Base) + `<out>/patch_embedding_lora.safetensors`.

Example (LoRA-17000, 368x640):
  python tools/merge_lora_partial_for_helios.py \
    --base /mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd \
    --base_subfolder transformer_init \
    --lora /mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368/checkpoint-17000/pytorch_lora_weights.safetensors \
    --partial /mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368/checkpoint-17000/transformer_partial.pth \
    --out /mnt/beegfs/xiangbo/helios_runs/_merged/stage1_lora368_merged17000 \
    --lora_alpha 128 --lora_rank 128
"""

import argparse
import os
import sys
from argparse import Namespace

import torch
from safetensors.torch import load_file, save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.utils.utils_base import load_extra_components


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base transformer dir (e.g. Helios-Base snapshot)")
    p.add_argument("--base_subfolder", default="transformer_init")
    p.add_argument("--lora", required=True, help="pytorch_lora_weights.safetensors of the LoRA checkpoint")
    p.add_argument("--partial", required=True, help="transformer_partial.pth of the LoRA checkpoint")
    p.add_argument("--out", required=True, help="output dir; writes transformer/ + patch_embedding_lora.safetensors")
    p.add_argument("--lora_alpha", type=float, default=128.0)
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--skip_module", default="patch_embedding", help="substring of modules to NOT merge")
    return p.parse_args()


def main():
    args = parse_args()
    scaling = args.lora_alpha / args.lora_rank
    print(f"[merge] scaling = alpha/rank = {args.lora_alpha}/{args.lora_rank} = {scaling}")

    # 1. build the base transformer (plain, no PEFT). kwargs mirror train_helios.py's stage-1 defaults.
    transformer_additional_kwargs = {
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
    print(f"[merge] loading base transformer: {args.base} (subfolder={args.base_subfolder})")
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.base,
        subfolder=args.base_subfolder,
        transformer_additional_kwargs=transformer_additional_kwargs,
    ).to(torch.float32)
    transformer.eval()
    name2mod = dict(transformer.named_modules())

    # 2. group the LoRA safetensors into per-module {A, B}, skipping patch_embedding.
    sd = load_file(args.lora)
    patch_embedding_keys = {}
    mods = {}
    for k, v in sd.items():
        if not k.startswith("transformer."):
            raise ValueError(f"unexpected key without 'transformer.' prefix: {k}")
        if args.skip_module in k:
            patch_embedding_keys[k] = v
            continue
        body = k[len("transformer."):]  # e.g. blocks.0.attn1.to_k.lora_A.weight
        if ".lora_A.weight" in body:
            mpath, ab = body[: -len(".lora_A.weight")], "A"
        elif ".lora_B.weight" in body:
            mpath, ab = body[: -len(".lora_B.weight")], "B"
        else:
            raise ValueError(f"unexpected LoRA key (no lora_A/lora_B): {k}")
        mods.setdefault(mpath, {})[ab] = v

    print(f"[merge] merging {len(mods)} Linear LoRA modules; keeping {len(patch_embedding_keys)} "
          f"'{args.skip_module}' keys as a separate adapter")

    # 3. bake delta into each merged module's weight: W += scaling * (B @ A).
    merged, missing = 0, []
    for mpath, ab in mods.items():
        if "A" not in ab or "B" not in ab:
            raise ValueError(f"module {mpath} missing an A/B half: {list(ab)}")
        if mpath not in name2mod:
            missing.append(mpath)
            continue
        module = name2mod[mpath]
        if not hasattr(module, "weight") or module.weight is None:
            raise ValueError(f"module {mpath} has no .weight to merge into ({type(module)})")
        A = ab["A"].to(torch.float32)  # [r, in]
        B = ab["B"].to(torch.float32)  # [out, r]
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"expected 2D Linear LoRA for {mpath}, got A{tuple(A.shape)} B{tuple(B.shape)}")
        delta = (B @ A) * scaling  # [out, in]
        w = module.weight
        if tuple(delta.shape) != tuple(w.shape):
            raise ValueError(f"shape mismatch for {mpath}: delta{tuple(delta.shape)} vs weight{tuple(w.shape)}")
        with torch.no_grad():
            w.add_(delta.to(w.dtype))
        merged += 1
    if missing:
        raise ValueError(f"{len(missing)} LoRA modules not found in the base transformer, e.g. {missing[:5]}")
    print(f"[merge] merged {merged} modules into base weights")

    # 4. overwrite the fully-trained memory patches (patch_short/mid/long) from transformer_partial.pth.
    #    is_train_full_patch_embedding=False -> load_extra_components will NOT touch patch_embedding.
    ec_args = Namespace()
    ec_args.training_config = Namespace(
        is_enable_stage1=True,
        is_train_full_patch_embedding=False,
        restrict_self_attn=False,
        is_amplify_history=False,
        is_use_gan=False,
    )
    print(f"[merge] loading memory patches from {args.partial}")
    load_extra_components(ec_args, transformer, args.partial)

    # 5. save the consolidated transformer.
    out_transformer = os.path.join(args.out, "transformer")
    os.makedirs(out_transformer, exist_ok=True)
    print(f"[merge] saving consolidated transformer -> {out_transformer}")
    transformer.to(torch.bfloat16).save_pretrained(out_transformer)

    # 6. save the patch_embedding LoRA verbatim (keys kept exactly as in pytorch_lora_weights).
    out_pe = os.path.join(args.out, "patch_embedding_lora.safetensors")
    save_file(patch_embedding_keys, out_pe)
    print(f"[merge] saved patch_embedding LoRA ({len(patch_embedding_keys)} tensors) -> {out_pe}")
    for k in patch_embedding_keys:
        print(f"          {k} {tuple(patch_embedding_keys[k].shape)}")
    print("[merge] DONE")


if __name__ == "__main__":
    main()

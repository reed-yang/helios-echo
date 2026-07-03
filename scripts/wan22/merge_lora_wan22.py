#!/usr/bin/env python
"""Merge a Stage-1 LoRA (+ trained multi-term-memory patch in transformer_partial.pth) into the
Wan2.2-TI2V-5B base, producing a standalone transformer/ for the next stage (post) or for eval.

Parameterized version of tools/merge_lora_for_helios.py with Stage-1 flags (no GAN / amplify).

  python scripts/wan22/merge_lora_wan22.py \
    --base /mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers \
    --lora    /mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22/checkpoint-5500/pytorch_lora_weights.safetensors \
    --partial /mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22/checkpoint-5500/transformer_partial.pth \
    --out     /mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22_merged
"""
import argparse
import os
from argparse import Namespace

from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.utils.utils_base import load_extra_components

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="transformer base the LoRA was trained on (init: Wan2.2 dir; post: merged-init dir)")
ap.add_argument("--pipe_base", default=None, help="dir providing vae/text_encoder/tokenizer (always Wan2.2); defaults to --base")
ap.add_argument("--lora", required=True, help="pytorch_lora_weights.safetensors from the trained checkpoint")
ap.add_argument("--partial", required=True, help="transformer_partial.pth (trained multi-term patch + norms)")
ap.add_argument("--out", required=True, help="output dir; writes <out>/transformer")
ap.add_argument("--base_subfolder", default="transformer")
args_cli = ap.parse_args()
pipe_base = args_cli.pipe_base or args_cli.base

transformer_additional_kwargs = {
    "has_multi_term_memory_patch": True,
    "zero_history_timestep": True,
    "guidance_cross_attn": True,
    "restrict_self_attn": False,
    "is_train_restrict_lora": False,
    "restrict_lora": False,
    "restrict_lora_rank": 128,
}

print(f"[merge] base transformer <- {args_cli.base}/{args_cli.base_subfolder}", flush=True)
transformer = HeliosTransformer3DModel.from_pretrained(
    args_cli.base,
    subfolder=args_cli.base_subfolder,
    transformer_additional_kwargs=transformer_additional_kwargs,
)
print(f"[merge] pipeline vae/text/tokenizer <- {pipe_base}", flush=True)
pipe = HeliosPipeline.from_pretrained(pipe_base, transformer=transformer)

print(f"[merge] LoRA <- {args_cli.lora}", flush=True)
pipe.load_lora_weights(args_cli.lora, adapter_name="default")
pipe.set_adapters(["default"], adapter_weights=[1.0])

# Load the trained non-LoRA trainable params (multi-term-memory patch + norm layers). Stage-1 flags only.
args = Namespace()
args.training_config = Namespace()
args.training_config.is_enable_stage1 = True
args.training_config.restrict_self_attn = False
args.training_config.is_amplify_history = False
args.training_config.is_use_gan = False
print(f"[merge] extra components (multi-term patch/norms) <- {args_cli.partial}", flush=True)
load_extra_components(args, transformer, args_cli.partial)

pipe.fuse_lora()
pipe.unload_lora_weights()
out_tf = os.path.join(args_cli.out, "transformer")
os.makedirs(args_cli.out, exist_ok=True)
print(f"[merge] saving merged transformer -> {out_tf}", flush=True)
pipe.transformer.save_pretrained(out_tf)
print("[merge] DONE", flush=True)

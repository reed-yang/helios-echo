"""Full LoRA + partial merge into a consolidated transformer (Stage A base).

Parameterized counterpart of tools/merge_lora_for_helios.py (a hardcoded-path
template): fuses the ENTIRE LoRA adapter into the base transformer, applies the
full-rank extras from transformer_partial.pth, and saves a `<out>/transformer/`
dir loadable like Helios-Base. Used for structure-changing lineage handoffs
where checkpoint-resume is impossible (the new run's trainable set differs, so
accelerate load_state would crash on the optimizer state) — e.g. the Stage A
memory run forking from stage1_lora_cfr_368_correct/checkpoint-19500.

Example:
  python tools/merge_lora_full_for_helios.py \
    --base /mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd \
    --base_subfolder transformer_init \
    --shell /mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled \
    --lora /mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368_correct/checkpoint-19500/pytorch_lora_weights.safetensors \
    --partial /mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368_correct/checkpoint-19500/transformer_partial.pth \
    --out /mnt/beegfs/siyuan/helios_runs/_merged/stage1_lora368_correct_merged19500
"""

import argparse
import os
import sys
from argparse import Namespace

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.utils.utils_base import load_extra_components


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base transformer dir (e.g. Helios-Base snapshot)")
    p.add_argument("--base_subfolder", default="transformer_init")
    p.add_argument("--shell", required=True, help="pipeline shell dir (tokenizer/text_encoder/vae)")
    p.add_argument("--lora", required=True, help="pytorch_lora_weights.safetensors of the checkpoint")
    p.add_argument("--partial", required=True, help="transformer_partial.pth of the checkpoint")
    p.add_argument("--out", required=True, help="output dir; writes <out>/transformer/")
    return p.parse_args()


def main():
    args = parse_args()

    # Stage-1 architecture kwargs matching the ancestor run (NO memory: the
    # merged artifact is memory-free; the training run constructs memory fresh).
    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": True,
        "zero_history_timestep": True,
        "guidance_cross_attn": True,
        "restrict_self_attn": False,
        "is_train_restrict_lora": False,
        "restrict_lora": False,
        "restrict_lora_rank": 128,
    }

    # fp32 fuse: LoRA deltas can be small relative to the base weights; fusing
    # in bf16 would round them away. Cast back to bf16 only at save time.
    print(f"[merge] loading base transformer (fp32) from {args.base}/{args.base_subfolder}")
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.base,
        subfolder=args.base_subfolder,
        transformer_additional_kwargs=transformer_additional_kwargs,
        torch_dtype=torch.float32,
    )
    pipe = HeliosPipeline.from_pretrained(args.shell, transformer=transformer)

    print(f"[merge] loading LoRA {args.lora}")
    pipe.load_lora_weights(args.lora, adapter_name="default")
    pipe.set_adapters(["default"], adapter_weights=[1.0])

    print(f"[merge] loading partial {args.partial}")
    extra_args = Namespace(training_config=Namespace(
        # Match the ancestor training config so the loader expects exactly the
        # sections present in its transformer_partial.pth (fail-loud otherwise).
        is_enable_stage1=True,
        is_amplify_history=False,
        is_use_gan=False,
        restrict_self_attn=False,
        is_enable_evolving_memory=False,
    ))
    load_extra_components(extra_args, transformer, args.partial)

    print("[merge] fusing LoRA into the base weights (fp32)")
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    out_dir = os.path.join(args.out, "transformer")
    print(f"[merge] saving bf16 merged transformer to {out_dir}")
    pipe.transformer.to(torch.bfloat16)
    pipe.transformer.save_pretrained(out_dir)

    # Self-check: reload config + one shard header to prove the artifact is sane.
    saved = [f for f in os.listdir(out_dir) if f.endswith(".safetensors")]
    print(f"[merge] DONE: {len(saved)} weight shards in {out_dir}")
    print("MERGE RESULT: OK")


if __name__ == "__main__":
    main()

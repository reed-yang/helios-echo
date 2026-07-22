"""P1 A1 smoke: assemble the training-lineage LoRA + partial checkpoint.

Decision record: docs/specs/2026-07-22-real-model-test-checkpoint-decision.md.
Run on one H200:
  srun -p gpu --gres=gpu:h200:1 --time=00:30:00 bash -c \
    "PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python tests/smoke_a1_assembly.py"
"""

import json
import sys
from types import SimpleNamespace

import torch
from safetensors import safe_open

from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.utils.utils_base import load_extra_components

BASE = "/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd"
CHECKPOINT = "/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368_correct/checkpoint-19500"
LORA_PATH = f"{CHECKPOINT}/pytorch_lora_weights.safetensors"
PARTIAL_PATH = f"{CHECKPOINT}/transformer_partial.pth"
MEM_KWARGS = dict(is_enable_evolving_memory=True, is_amplify_memory=True)
DEV = "cuda"

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def make_inputs(dtype):
    torch.manual_seed(0)
    # 384x640: latent 48x80; pipeline id layout x0=0, long 1..16, mid 17-18,
    # newest 19, current 20..28 (pipeline_helios.py:1265-1298 semantics).
    def r(*shape):
        return torch.randn(*shape, device=DEV, dtype=dtype)

    return dict(
        hidden_states=r(1, 16, 9, 48, 80),
        timestep=torch.tensor([500.0], device=DEV),
        encoder_hidden_states=r(1, 512, 4096),
        indices_hidden_states=torch.arange(20.0, 29.0, device=DEV).unsqueeze(0),
        indices_latents_history_short=torch.tensor([[0.0, 19.0]], device=DEV),
        indices_latents_history_mid=torch.tensor([[17.0, 18.0]], device=DEV),
        indices_latents_history_long=torch.arange(1.0, 17.0, device=DEV).unsqueeze(0),
        latents_history_short=r(1, 16, 2, 48, 80),
        latents_history_mid=r(1, 16, 2, 48, 80),
        latents_history_long=r(1, 16, 16, 48, 80),
        return_dict=False,
    )


def extra_component_args():
    # Source recipe flags: stage1_lora_cfr_368_correct.yaml:135,173,176,179-180,
    # 183-185. GAN is absent there and defaults false (train_config.py:365).
    return SimpleNamespace(
        training_config=SimpleNamespace(
            is_enable_stage1=True,
            is_train_full_patch_embedding=False,
            is_train_full_multi_term_memory_patchg=True,
            is_train_lora_multi_term_memory_patchg=False,
            is_amplify_history=False,
            restrict_self_attn=False,
            is_train_restrict_lora=False,
            restrict_lora=False,
            is_use_gan=False,
            is_enable_evolving_memory=False,
        )
    )


def checkpoint_keys(partial_state):
    index_path = f"{BASE}/transformer_init/diffusion_pytorch_model.safetensors.index.json"
    with open(index_path, encoding="utf-8") as f:
        base_keys = set(json.load(f)["weight_map"])
    with safe_open(LORA_PATH, framework="pt", device="cpu") as f:
        lora_keys = set(f.keys())
    return base_keys | lora_keys | set(partial_state)


def count_loaded_partial_keys(transformer, partial_state):
    loaded = 0
    for key, expected in partial_state.items():
        module_name, local_key = key.split(".", 1)
        module = getattr(transformer, module_name, None)
        if module is None or local_key not in module.state_dict():
            continue
        actual = module.state_dict()[local_key]
        expected = expected.to(device=actual.device, dtype=actual.dtype)
        loaded += int(torch.equal(actual, expected))
    return loaded


def main():
    partial_state = torch.load(PARTIAL_PATH, map_location="cpu", weights_only=True)
    all_checkpoint_keys = checkpoint_keys(partial_state)

    transformer = HeliosTransformer3DModel.from_pretrained(
        BASE,
        subfolder="transformer_init",
        transformer_additional_kwargs=MEM_KWARGS,
        torch_dtype=torch.bfloat16,
        device_map=DEV,
    )
    transformer.eval()

    # Use the training-side pipeline mixin rather than recreating LoraConfig by
    # hand: it consumes the adapter metadata saved by the trainer and injects the
    # PEFT adapter into this exact training-side transformer, without loading VAE/T5.
    pipe = HeliosPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        scheduler=None,
        transformer=transformer,
    )
    pipe.load_lora_weights(CHECKPOINT, adapter_name="default")
    pipe.set_adapters(["default"], adapter_weights=[1.0])
    transformer = pipe.transformer

    lora_params = [(name, param) for name, param in transformer.named_parameters() if "lora_" in name]
    nonzero_lora_b = next(
        (
            name
            for name, param in lora_params
            if "lora_B" in name and bool(torch.count_nonzero(param.detach()).item())
        ),
        None,
    )
    check(
        "adapter applied with nonzero PEFT B weights",
        bool(lora_params) and nonzero_lora_b is not None,
        f"tensors={len(lora_params)} probe={nonzero_lora_b}",
    )

    load_extra_components(extra_component_args(), transformer, PARTIAL_PATH)
    loaded_partial = count_loaded_partial_keys(transformer, partial_state)
    check(
        "partial patch modules loaded",
        loaded_partial > 0,
        f"loaded={loaded_partial}/{len(partial_state)} keys",
    )

    memory_checkpoint_keys = sorted(
        key for key in all_checkpoint_keys if "evolving_memory" in key or "memory_key_scale" in key
    )
    memory_params = dict(transformer.named_parameters())
    check("fresh evolving_memory present", "evolving_memory.query_init" in memory_params)
    check(
        "memory absent from base, LoRA, and partial checkpoints",
        not memory_checkpoint_keys,
        str(memory_checkpoint_keys[:3]),
    )
    check("fresh memory state not initialized", transformer.evolving_memory.query_state is None)
    check(
        "adapter did not wrap evolving_memory",
        not any("evolving_memory" in name and "lora_" in name for name, _ in lora_params),
    )

    inputs = make_inputs(torch.bfloat16)
    with torch.inference_mode():
        out_plain = transformer(**inputs)[0]
    check("forward without memory tokens finite", bool(torch.isfinite(out_plain).all()))

    transformer.evolving_memory.reset(1, device=DEV)
    tokens = transformer.evolving_memory.get_tokens(dtype=torch.bfloat16)
    with torch.inference_mode():
        out_memory = transformer(**inputs, memory_tokens=tokens)[0]
    check("forward with memory tokens finite", bool(torch.isfinite(out_memory).all()))
    check(
        "paired forward output shapes equal",
        out_memory.shape == out_plain.shape,
        f"plain={tuple(out_plain.shape)} memory={tuple(out_memory.shape)}",
    )

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak GPU memory: {peak:.1f} GiB")
    print("SMOKE RESULT:", "FAILED " + str(failures) if failures else "ALL PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

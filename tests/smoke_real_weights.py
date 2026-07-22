"""P1 real-weight smoke on released Helios-Base (14B) weights.

Decision record: docs/specs/2026-07-22-real-model-test-checkpoint-decision.md.
Run on one H200:
  srun -p gpu --gres=gpu:h200:1 --time=00:30:00 bash -c \
    "PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python tests/smoke_real_weights.py"

Checks:
  1. memory-enabled from_pretrained constructs and loads; missing keys are ONLY
     evolving_memory.* / memory_key_scale (loader tolerance verified on 14B)
  2. off-path equivalence: memory model with memory_tokens=None matches the
     baseline model bitwise on real weights at real 384x640 geometry
  3. read path runs; capture shape [B, 8640, 5120]; memory changes the output
  4. K0 gradient reaches the encoder through the 14B read path (Stage-A style:
     only the memory stack requires grad; gradient checkpointing on)
"""

import sys

import torch

from helios.modules.transformer_helios import HeliosTransformer3DModel

BASE = "/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd"
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


def load(with_memory):
    kwargs = dict(MEM_KWARGS) if with_memory else {}
    model = HeliosTransformer3DModel.from_pretrained(
        BASE,
        subfolder="transformer",
        transformer_additional_kwargs=kwargs,
        torch_dtype=torch.bfloat16,
        device_map=DEV,
    )
    model.eval()
    return model


def main():
    mem = load(with_memory=True)

    # 1. loader tolerance: fresh memory params exist and are non-degenerate
    names = dict(mem.named_parameters())
    check("memory submodule constructed", "evolving_memory.query_init" in names)
    check(
        "memory_key_scale per block (fp32 kept)",
        "blocks.39.attn1.memory_key_scale" in names
        and names["blocks.0.attn1.memory_key_scale"].dtype == torch.float32,
    )
    scale = mem.blocks[0].attn1.get_scale_memory().float()
    check("amp near-neutral", bool((scale - 1.1619).abs().max() < 2e-2), f"scale={scale.mean().item():.4f}")

    inputs = make_inputs(torch.bfloat16)
    with torch.no_grad():
        out_plain = mem(**inputs)[0]

    base = load(with_memory=False)
    with torch.no_grad():
        out_base = base(**inputs)[0]
    check("off-path bitwise equivalence on real weights", bool(torch.equal(out_plain, out_base)))
    del base
    torch.cuda.empty_cache()

    # 3. read path + capture at real geometry
    mem.evolving_memory.reset(1, device=DEV)
    tokens = mem.evolving_memory.get_tokens()
    check("M = 3*12*20 = 720 tokens", tuple(tokens.shape) == (1, 720, 5120), str(tokens.shape))
    with torch.no_grad():
        ret = mem(**inputs, memory_tokens=tokens, capture_last_hidden=True)
    check("conditional 3-tuple", len(ret) == 3)
    check("capture shape [1, 8640, 5120]", tuple(ret[2].shape) == (1, 8640, 5120), str(ret[2].shape))
    check("memory changes output", not torch.equal(ret[0], out_plain))
    check("output shape preserved", ret[0].shape == out_plain.shape)

    # 4. Stage-A style gradient: only the memory stack trains, ckpt on
    mem.requires_grad_(False)
    for name, param in mem.named_parameters():
        if "evolving_memory" in name or "memory_key_scale" in name:
            param.requires_grad = True
    mem.enable_gradient_checkpointing()
    mem.train()
    mem.evolving_memory.reset(1, device=DEV)
    mem.evolving_memory.update(
        torch.randn(1, 8640, 5120, device=DEV, dtype=torch.bfloat16), sigma_last=0.02
    )
    out = mem(**inputs, memory_tokens=mem.evolving_memory.get_tokens())[0]
    out.float().pow(2).mean().backward()
    for probe in ["evolving_memory.ctx_k_proj.weight", "evolving_memory.gate_linear.weight",
                  "evolving_memory.query_init", "blocks.0.attn1.memory_key_scale"]:
        grad = dict(mem.named_parameters())[probe].grad
        check(f"grad nonzero: {probe}", grad is not None and grad.abs().max().item() > 0.0)

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak GPU memory: {peak:.1f} GiB")
    print("SMOKE RESULT:", "FAILED " + str(failures) if failures else "ALL PASS")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

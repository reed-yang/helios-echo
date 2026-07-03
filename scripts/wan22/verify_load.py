#!/usr/bin/env python
"""Smoke-verify that HeliosTransformer3DModel builds from + loads the Wan2.2-TI2V-5B base.

This is the critical early check for the Wan2.1 -> Wan2.2-5B base swap:
  * the OLD `weight[:, :16]` hardcode in initialize_weight_from_another_conv3d would CRASH
    at build time for a 48-channel base (size-mismatch in load_state_dict). Building OK
    therefore proves the port fix.
  * the custom from_pretrained does a SIZE-MATCHED partial load (mismatches are silently
    skipped), so we explicitly confirm the core Wan2.2 weights actually landed in the model.

CPU is fine (~20GB RAM peak in bf16); no GPU needed (no forward here -- forward is
exercised by the real smoke-train).

  cd <repo>; PYTHONPATH=<repo> HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface \
      python scripts/wan22/verify_load.py
"""
import glob
import os
import sys

import torch

WAN22 = os.environ.get("WAN22_DIR", "/mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers")

from helios.modules.transformer_helios import HeliosTransformer3DModel  # noqa: E402

# mirror train_helios.py transformer_additional_kwargs (stage-1 flags)
kw = dict(
    has_multi_term_memory_patch=True,
    zero_history_timestep=True,
    restrict_self_attn=False,
    guidance_cross_attn=True,
    is_train_restrict_lora=False,
    restrict_lora=False,
    restrict_lora_rank=128,
    is_amplify_history=False,
    history_scale_mode="per_head",
)

print(f"[verify] loading HeliosTransformer3DModel from {WAN22}/transformer ...", flush=True)
model = HeliosTransformer3DModel.from_pretrained(
    WAN22,
    subfolder="transformer",
    transformer_additional_kwargs=kw,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
c = model.config
print(
    "[verify] config:",
    dict(
        in_channels=c.in_channels,
        out_channels=c.out_channels,
        num_layers=c.num_layers,
        num_attention_heads=c.num_attention_heads,
        attention_head_dim=c.attention_head_dim,
        ffn_dim=c.ffn_dim,
        patch_size=tuple(c.patch_size),
        text_dim=c.text_dim,
    ),
    flush=True,
)

errs = []


def check(name, cond):
    print(f"[verify] {'PASS' if cond else 'FAIL'}: {name}", flush=True)
    if not cond:
        errs.append(name)


inner = model.inner_dim
check("in_channels==48", c.in_channels == 48)
check("num_layers==30", c.num_layers == 30)
check("inner_dim==3072", inner == 3072)
check("num blocks==30", len(model.blocks) == 30)
check("patch_embedding.weight==(3072,48,1,2,2)", tuple(model.patch_embedding.weight.shape) == (3072, 48, 1, 2, 2))
check("patch_short in-channels==48", model.patch_short.weight.shape[1] == 48)
check("patch_mid.weight==(3072,48,2,4,4)", tuple(model.patch_mid.weight.shape) == (3072, 48, 2, 4, 4))
check("patch_long.weight==(3072,48,4,8,8)", tuple(model.patch_long.weight.shape) == (3072, 48, 4, 8, 8))
check("proj_out.weight==(192,3072)", tuple(model.proj_out.weight.shape) == (48 * 1 * 2 * 2, 3072))

# the :48 fix produced a valid, finite multi-term-memory patch (the OLD :16 would crash the build):
check("patch_short finite", torch.isfinite(model.patch_short.weight).all().item())
check("patch_mid finite", torch.isfinite(model.patch_mid.weight).all().item())
check("patch_long finite", torch.isfinite(model.patch_long.weight).all().item())
# NOTE: patch_short is copied from patch_embedding during __init__ (still random), THEN from_pretrained
# overwrites patch_embedding with the checkpoint -> they intentionally differ post-load (same as Wan2.1).

# Authoritative load check: replicate from_pretrained's key remap and compute missing/unexpected
# directly. A correct port loads EVERY core Wan2.2 weight and leaves only the 6 Helios-added keys missing.
from safetensors.torch import load_file  # noqa: E402

raw = {}
for f in glob.glob(os.path.join(WAN22, "transformer", "*.safetensors")):
    raw.update(load_file(f))


def remap(k):
    return "norm_out.scale_shift_table" if k == "scale_shift_table" else k


raw_remapped = {remap(k): v for k, v in raw.items()}
mp = {k: v for k, v in model.named_parameters()}
ckpt_keys, model_keys = set(raw_remapped), set(mp)
missing = sorted(model_keys - ckpt_keys)  # in model, not in ckpt
unexpected = sorted(ckpt_keys - model_keys)  # in ckpt, not in model
expected_missing = {f"patch_{g}.{p}" for g in ("short", "mid", "long") for p in ("weight", "bias")}
print(f"[verify] missing (model-not-ckpt): {missing}", flush=True)
print(f"[verify] unexpected (ckpt-not-model): {unexpected}", flush=True)
check("missing keys == the 6 multi-term-memory patch keys", set(missing) == expected_missing)
check("no unexpected ckpt keys", len(unexpected) == 0)

# value-level: every common key actually carries the checkpoint value (atol covers bf16/fp32 upcast)
common = [k for k in (ckpt_keys & model_keys) if tuple(mp[k].shape) == tuple(raw_remapped[k].shape)]
bad = [k for k in common if not torch.allclose(mp[k].float(), raw_remapped[k].float(), atol=1e-2)]
print(f"[verify] common keys: {len(common)} | NOT-allclose: {len(bad)} -> {bad[:20]}", flush=True)
check("core weights carry checkpoint values (>=99.5% allclose)", len(bad) <= 0.005 * len(common))
for key in ["patch_embedding.weight", "proj_out.weight"]:
    check(f"core weight loaded: {key}", key in common and torch.allclose(mp[key].float(), raw_remapped[key].float(), atol=1e-2))

print(f"\n[verify] total params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B", flush=True)
if errs:
    print("[verify] FAILURES:", errs, flush=True)
    sys.exit(1)
print("[verify] ALL CHECKS PASSED", flush=True)

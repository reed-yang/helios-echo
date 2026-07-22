"""End-to-end rollout smoke: memory state machine over 5 sections, real weights.

Verifies the pipeline-state-machine plan's GPU gate: first eviction write at
k=2 (three writes over 5 sections), queue drains to 2 pending entries, state
export works, generation stays finite, and the memory overhead vs a no-memory
run of the same shape is measured.

Run on one H200 (xiangbo env):
  srun -p gpu --gres=gpu:h200:1 --time=01:40:00 bash -c \
    "PYTHONPATH=. /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python tests/smoke_rollout.py"
"""

import sys
import time

import torch

from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline

BASE = "/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd"
MEM_KWARGS = dict(is_enable_evolving_memory=True, is_amplify_memory=True)
PROMPT = "A red vintage car drives along a coastal road at sunset, waves crashing on the rocks."
# Validated Base sampling regime (root-cause report 2026-07-22: 8-step + fixed
# mu=1 shift + empty negative prompt NaNs; both known-good entries use 50 steps
# + resolution-dependent exponential shifting + the standard negative prompt).
NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
GEN_KWARGS = dict(
    negative_prompt=NEGATIVE_PROMPT,
    height=384,
    width=640,
    num_frames=165,  # 5 sections x 33 RGB frames
    num_inference_steps=50,
    guidance_scale=5.0,
    use_dynamic_shifting=True,
    time_shift_type="exponential",
    output_type="latent",
)

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def main():
    transformer = HeliosTransformer3DModel.from_pretrained(
        BASE,
        subfolder="transformer",
        transformer_additional_kwargs=dict(MEM_KWARGS),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    pipe = HeliosPipeline.from_pretrained(BASE, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Memory-on rollout
    torch.manual_seed(7)
    start = time.time()
    out_mem = pipe(prompt=PROMPT, enable_evolving_memory=True, **GEN_KWARGS)
    t_mem = time.time() - start

    state = pipe.get_memory_state()
    check("memory state exported", state is not None)
    if state is not None:
        m0 = transformer.evolving_memory.query_init.detach().float().cpu().expand_as(state["M"])
        check("state evolved from M0 (3 writes over 5 sections)", not torch.allclose(state["M"], m0))
        check("queue holds 2 pending sections", len(state["queue"]) == 2, f"len={len(state['queue'])}")
        if len(state["queue"]) == 2:
            check("pending sections are 3 and 4", [e[0] for e in state["queue"]] == [3, 4])
            sigmas = {e[2] for e in state["queue"]}
            check("per-capture sigma recorded", all(s is not None for s in sigmas), str(sigmas))
    latents = out_mem.frames if hasattr(out_mem, "frames") else out_mem[0]
    if torch.is_tensor(latents):
        check("memory-on latents finite", bool(torch.isfinite(latents).all()))

    # No-memory rollout, same seed/shape (overhead + regression reference)
    torch.manual_seed(7)
    start = time.time()
    out_plain = pipe(prompt=PROMPT, enable_evolving_memory=False, **GEN_KWARGS)
    t_plain = time.time() - start
    latents_plain = out_plain.frames if hasattr(out_plain, "frames") else out_plain[0]
    if torch.is_tensor(latents_plain):
        check("no-memory latents finite", bool(torch.isfinite(latents_plain).all()))

    overhead = (t_mem - t_plain) / t_plain * 100 if t_plain > 0 else float("nan")
    print(f"wall time: memory-on {t_mem:.1f}s, no-memory {t_plain:.1f}s, overhead {overhead:+.1f}%", flush=True)
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print("SMOKE ROLLOUT RESULT:", "FAILED " + str(failures) if failures else "ALL PASS", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

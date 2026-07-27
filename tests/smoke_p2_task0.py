"""P2 Task 0 smoke for Distilled weights through the training-side pipeline.

This is the pre-scale gate from
``docs/plans/2026-07-22-p2-interim-drift-ab-plan.md``. It first runs a
one-section latent-only generation, then (unless ``--skip-dual`` is given) a
paired memory-on/no-KV rollout in the mapped released Distilled regime.

Run the one-section gate first on one H200::

    PYTHONPATH=. python tests/smoke_p2_task0.py --skip-dual

Then run the default five-section dual-arm gate::

    PYTHONPATH=. python tests/smoke_p2_task0.py
"""

import argparse
from pathlib import Path


DISTILLED = Path("/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled")
PROMPT = (
    "A vibrant tropical fish swimming gracefully among colorful coral reefs "
    "in a clear, turquoise ocean. The fish has bright blue and yellow scales "
    "with a small, distinctive orange spot on its side, its fins moving "
    "fluidly. The coral reefs are alive with a variety of marine life, "
    "including small schools of colorful fish and sea turtles gliding by."
)
MEMORY_KEY_MARKERS = ("evolving_memory", "memory_key_scale")
MEMORY_KWARGS = {
    "is_enable_evolving_memory": True,
    "is_amplify_memory": True,
}
SEED = 7


class Checks:
    def __init__(self, total):
        self.total = total
        self.current = 0

    def require(self, condition, name, detail=""):
        self.current += 1
        suffix = f" ({detail})" if detail else ""
        if condition:
            print(
                f"CHECK {self.current}/{self.total} PASS: {name}{suffix}",
                flush=True,
            )
            return
        print(
            f"CHECK {self.current}/{self.total} FAIL: {name}{suffix}",
            flush=True,
        )
        raise AssertionError(name)


class MethodProbe:
    """Temporarily wrap an instance method and retain observations."""

    def __init__(self, instance, method_name, wrapper_factory):
        self.instance = instance
        self.method_name = method_name
        self.original = getattr(instance, method_name)
        self.wrapper = wrapper_factory(self.original)

    def __enter__(self):
        setattr(self.instance, self.method_name, self.wrapper)
        return self.wrapper

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.instance, self.method_name, self.original)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sections",
        type=int,
        default=5,
        help="number of sections in each dual-arm rollout (default: 5)",
    )
    parser.add_argument(
        "--skip-dual",
        action="store_true",
        help="run only the mandatory one-section gate",
    )
    args = parser.parse_args()
    if not args.skip_dual and args.sections < 3:
        parser.error("--sections must be at least 3 for write-count checks")
    return args


def checkpoint_memory_keys():
    from safetensors import safe_open

    hits = []
    transformer_dir = DISTILLED / "transformer"
    shard_paths = sorted(transformer_dir.glob("*.safetensors"))
    if not shard_paths:
        raise AssertionError(f"no safetensors shards found in {transformer_dir}")
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            hits.extend(
                f"{shard_path.name}:{key}"
                for key in shard.keys()
                if any(marker in key for marker in MEMORY_KEY_MARKERS)
            )
    return shard_paths, hits


def generation_kwargs(num_sections, scheduler_config):
    # Exact training-side mapping of scripts/inference/helios-distilled_t2v.sh
    # plus the scheduler-config defaults consumed implicitly by the released
    # diffusers pipeline. Guidance 1.0 disables negative-prompt encoding.
    return {
        "prompt": PROMPT,
        "height": 384,
        "width": 640,
        "num_frames": num_sections * 33,
        "output_type": "latent",
        "guidance_scale": 1.0,
        "use_dmd": True,
        "is_enable_stage2": True,
        "stage2_num_inference_steps_list": [2, 2, 2],
        "stage2_num_stages": 3,
        "is_amplify_first_chunk": True,
        "use_dynamic_shifting": scheduler_config.use_dynamic_shifting,
        "time_shift_type": scheduler_config.time_shift_type,
    }


def output_latents(output):
    return output.frames if hasattr(output, "frames") else output[0]


def expected_latent_shape(num_sections):
    return (1, 16, num_sections * 9, 48, 80)


def latent_detail(torch, latents):
    if not torch.is_tensor(latents):
        return str(type(latents))
    if latents.numel() == 0:
        return f"shape={tuple(latents.shape)} numel=0"
    values = latents.detach().float()
    return (
        f"shape={tuple(latents.shape)} mean={values.mean().item():.6g} "
        f"std={values.std(unbiased=False).item():.6g} "
        f"absmax={values.abs().max().item():.6g}"
    )


def valid_latents(torch, latents, num_sections):
    if not torch.is_tensor(latents):
        return False
    if tuple(latents.shape) != expected_latent_shape(num_sections) or latents.numel() == 0:
        return False
    values = latents.detach().float()
    return (
        bool(torch.isfinite(values).all())
        and bool(torch.count_nonzero(values))
        and values.std(unbiased=False).item() > 0.0
    )


def seeded_generator(torch, seed):
    torch.manual_seed(seed)
    return torch.Generator(device="cuda").manual_seed(seed)


def load_training_pipeline(torch):
    from helios.modules.transformer_helios import HeliosTransformer3DModel
    from helios.pipelines.pipeline_helios import HeliosPipeline
    from helios.scheduler.scheduling_helios import HeliosScheduler

    transformer = HeliosTransformer3DModel.from_pretrained(
        str(DISTILLED),
        subfolder="transformer",
        transformer_additional_kwargs=dict(MEMORY_KWARGS),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    scheduler = HeliosScheduler.from_pretrained(
        str(DISTILLED),
        subfolder="scheduler",
        stages=3,
    )
    pipe = HeliosPipeline.from_pretrained(
        str(DISTILLED),
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    return pipe, transformer, HeliosPipeline


def main():
    args = parse_args()

    # Keep imports and all model construction under main so importing this smoke
    # script remains a CPU-only operation and never initializes CUDA.
    import torch

    total_checks = 9 if args.skip_dual else 17
    checks = Checks(total_checks)

    checkpoint_shards, memory_checkpoint_keys = checkpoint_memory_keys()
    checks.require(
        not memory_checkpoint_keys,
        "all Distilled checkpoint shards contain no evolving-memory tensors",
        f"shards={len(checkpoint_shards)} hits={memory_checkpoint_keys[:3]}",
    )

    pipe, transformer, pipeline_class = load_training_pipeline(torch)
    memory = getattr(transformer, "evolving_memory", None)
    checks.require(memory is not None, "memory module is present after checkpoint load")
    checks.require(
        memory is not None and memory.query_state is None,
        "checkpoint load leaves query_state untouched",
    )

    memory.reset(1, device="cuda")
    expected_m0 = memory.query_init.detach().float().expand_as(memory.query_state)
    checks.require(
        torch.equal(memory.query_state, expected_m0),
        "query_state starts at exact M0",
    )
    checks.require(
        isinstance(pipe, pipeline_class) and pipe.transformer is transformer,
        "Distilled model uses the training-side HeliosPipeline",
    )
    checks.require(
        pipe.scheduler.config.use_dynamic_shifting is True
        and pipe.scheduler.config.time_shift_type == "linear",
        "Distilled scheduler enables released linear dynamic shifting",
        (
            f"use_dynamic_shifting={pipe.scheduler.config.use_dynamic_shifting} "
            f"time_shift_type={pipe.scheduler.config.time_shift_type}"
        ),
    )
    proj_out = transformer.proj_out.weight.detach().float()
    checks.require(
        proj_out.numel() > 0
        and bool(torch.isfinite(proj_out).all())
        and bool(torch.count_nonzero(proj_out)),
        "checkpoint load restores nonzero proj_out weights",
        f"shape={tuple(proj_out.shape)} absmax={proj_out.abs().max().item():.6g}",
    )

    one_output = pipe(
        generator=seeded_generator(torch, SEED),
        enable_evolving_memory=True,
        **generation_kwargs(1, pipe.scheduler.config),
    )
    one_latents = output_latents(one_output)
    checks.require(
        torch.is_tensor(one_latents)
        and tuple(one_latents.shape) == expected_latent_shape(1)
        and one_latents.numel() > 0,
        "one-section generation returns the exact latent-only shape",
        latent_detail(torch, one_latents),
    )
    checks.require(
        valid_latents(torch, one_latents, 1),
        "one-section memory-on latents are finite and non-degenerate",
        latent_detail(torch, one_latents),
    )

    if args.skip_dual:
        print("P2 TASK 0 SMOKE: ALL PASS", flush=True)
        return

    write_sigmas = []

    def wrap_update(original):
        def counted_update(evicted_hidden, frame_mask=None, sigma_last=None):
            write_sigmas.append(sigma_last)
            return original(
                evicted_hidden,
                frame_mask=frame_mask,
                sigma_last=sigma_last,
            )

        return counted_update

    with MethodProbe(memory, "update", wrap_update):
        on_output = pipe(
            generator=seeded_generator(torch, SEED),
            enable_evolving_memory=True,
            **generation_kwargs(args.sections, pipe.scheduler.config),
        )
    on_latents = output_latents(on_output)
    checks.require(
        valid_latents(torch, on_latents, args.sections),
        "dual-arm memory-on latents have exact shape and are finite and non-degenerate",
        latent_detail(torch, on_latents),
    )
    on_state = pipe.get_memory_state()

    off_memory_tokens = []

    def wrap_forward(original):
        def tracked_forward(*forward_args, **forward_kwargs):
            off_memory_tokens.append(forward_kwargs.get("memory_tokens"))
            return original(*forward_args, **forward_kwargs)

        return tracked_forward

    with MethodProbe(transformer, "forward", wrap_forward):
        off_output = pipe(
            generator=seeded_generator(torch, SEED),
            enable_evolving_memory=False,
            **generation_kwargs(args.sections, pipe.scheduler.config),
        )
    off_latents = output_latents(off_output)
    checks.require(
        valid_latents(torch, off_latents, args.sections),
        "dual-arm memory-off latents have exact shape and are finite and non-degenerate",
        latent_detail(torch, off_latents),
    )
    checks.require(
        bool(off_memory_tokens) and all(tokens is None for tokens in off_memory_tokens),
        "memory-off arm is true no-KV",
        f"forward_calls={len(off_memory_tokens)}",
    )
    checks.require(on_state is not None, "memory-on state exports after rollout")

    expected_writes = args.sections - 2
    checks.require(
        len(write_sigmas) == expected_writes,
        "memory-on write count equals sections - 2",
        f"writes={len(write_sigmas)} expected={expected_writes}",
    )

    queue = on_state["queue"]
    expected_queue_indices = list(range(args.sections - 2, args.sections))
    checks.require(
        [entry[0] for entry in queue] == expected_queue_indices,
        "memory queue retains the final two sections",
        f"indices={[entry[0] for entry in queue]}",
    )

    all_capture_sigmas = write_sigmas + [entry[2] for entry in queue]
    checks.require(
        len(all_capture_sigmas) == args.sections
        and all(sigma is not None for sigma in all_capture_sigmas),
        "every section retains its own capture sigma",
        str(all_capture_sigmas),
    )
    checks.require(
        bool(torch.isfinite(on_state["M"]).all())
        and not torch.equal(on_state["M"], expected_m0.detach().cpu()),
        "memory state is finite and evolved from M0",
    )

    print(
        f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
        flush=True,
    )
    print("P2 TASK 0 SMOKE: ALL PASS", flush=True)


if __name__ == "__main__":
    main()

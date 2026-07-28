#!/usr/bin/env python3
"""Run paired P2 interim drift rollouts through the training-side pipeline.

Each non-dry-run invocation owns exactly one ``(prompt, arm)`` artifact. Launch
matching ``on`` and ``off`` invocations with the same ``--run-id``,
``--base-seed``, and ``--prompt-index`` to form a pair.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISTILLED = Path("/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled")
VS24_CONFIG = PROJECT_ROOT / "tools/long_video_eval/configs/vs24_long.json"
REP50_DIR = Path("/mnt/beegfs/xiangbo/helios_runs/eval_prompts_rep50")
REP50_FIRST_CASE = 3000
RESULTS_ROOT = PROJECT_ROOT / "results/p2_interim"
MEMORY_KWARGS = {
    "is_enable_evolving_memory": True,
    "is_amplify_memory": True,
}
REGIME = {
    "height": 384,
    "width": 640,
    "fps": 24,
    "rgb_frames_per_section": 33,
    "latent_frames_per_section": 9,
    "guidance_scale": 1.0,
    "use_dmd": True,
    "is_enable_stage2": True,
    "stage2_num_inference_steps_list": [2, 2, 2],
    "stage2_num_stages": 3,
    "is_amplify_first_chunk": True,
    "use_dynamic_shifting": True,
    "time_shift_type": "linear",
    "video_codec": "libx264",
}


class Milestones:
    def __init__(self):
        self.number = 0

    def emit(self, message):
        self.number += 1
        print(f"MILESTONE {self.number:03d} {message}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-index",
        type=int,
        help="zero-based index within the selected vs24_long prompts",
    )
    parser.add_argument(
        "--arm",
        choices=("on", "off"),
        help="memory arm; on enables evolving memory, off is true no-KV",
    )
    parser.add_argument(
        "--gpu",
        "--gpu-selectable",
        dest="gpu",
        help="CUDA device selector written to CUDA_VISIBLE_DEVICES before torch import",
    )
    parser.add_argument(
        "--sections",
        type=int,
        default=66,
        help="number of 33-frame sections (default: 66 = 2178 frames)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=2,
        help="number of leading prompts in this A/B campaign",
    )
    parser.add_argument(
        "--prompt-set",
        choices=("vs24_long", "rep50"),
        default="vs24_long",
        help=(
            "prompt source: vs24_long = raw sentences (legacy r1/r2 protocol); "
            "rep50 = structurally rewritten standard cases (segment 0 of each case CSV)"
        ),
    )
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument(
        "--memory-partial",
        type=Path,
        help=(
            "optional transformer_partial.pth from Stage A training; loads trained "
            "memory components (evolving_memory, patch convs, memory_key_scale) into "
            "the freshly constructed transformer; on-arm only"
        ),
    )
    parser.add_argument(
        "--run-id",
        help="shared A/B run identifier (pass the same value to parallel processes)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT,
        help="artifact parent directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve prompts, seeds, regime, and paths without importing torch",
    )
    args = parser.parse_args(argv)

    if args.sections < 1:
        parser.error("--sections must be positive")
    if args.prompt_index is not None and not 0 <= args.prompt_index < args.num_prompts:
        parser.error("--prompt-index must be within [0, --num-prompts)")
    if not args.dry_run and (args.prompt_index is None or args.arm is None):
        parser.error("a real rollout requires both --prompt-index and --arm")
    if not args.dry_run and args.run_id is None:
        parser.error("a real rollout requires --run-id shared by every paired process")
    if args.memory_partial is not None:
        if args.arm != "on":
            parser.error("--memory-partial is only valid with --arm on")
        if not args.memory_partial.is_file():
            parser.error(f"--memory-partial not found: {args.memory_partial}")
    return args


def utc_run_id():
    return datetime.now(timezone.utc).strftime("p2-%Y%m%dT%H%M%SZ")


def validate_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("--run-id may contain only letters, digits, '.', '_', and '-'")
    return run_id


def load_vs24_prompts(num_prompts):
    config = json.loads(VS24_CONFIG.read_text())
    prompt_path = Path(config["defaults"]["prompt_file"])
    prompts = [line.strip() for line in prompt_path.read_text().splitlines() if line.strip()]
    if len(prompts) < num_prompts:
        raise ValueError(f"{prompt_path} contains {len(prompts)} prompts; need {num_prompts}")
    return prompt_path, prompts[:num_prompts]


def load_rep50_prompts(num_prompts):
    """Segment 0 of rep50 standard cases 3000..3000+num-1 (structural rewrite).

    Each case CSV holds a multi-segment event sequence sharing one identity and
    background; the single-prompt drift protocol uses the first segment only.
    Returns per-index source paths so each manifest can hash its exact case file.
    """
    sources, prompts = [], []
    for case in range(REP50_FIRST_CASE, REP50_FIRST_CASE + num_prompts):
        path = REP50_DIR / f"{case}.csv"
        with path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["prompt_index"] == "0"]
        if len(rows) != 1:
            raise ValueError(f"{path} has {len(rows)} rows with prompt_index=0; expected exactly 1")
        sources.append(path)
        prompts.append(rows[0]["prompt"].strip())
    return sources, prompts


def paired_seed(base_seed, prompt_index):
    # Arm is deliberately absent so both processes construct the same RNG stream.
    return base_seed + prompt_index


def canonical_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def artifact_paths(output_root, run_id, arm, prompt_index):
    stem = f"prompt_{prompt_index:02d}"
    arm_dir = output_root.resolve() / run_id / arm
    return {
        "video": arm_dir / f"{stem}.mp4",
        "manifest": arm_dir / f"{stem}.manifest.json",
    }


def resolved_entries(args, run_id, prompts):
    prompt_indices = range(args.num_prompts) if args.prompt_index is None else [args.prompt_index]
    arms = ("on", "off") if args.arm is None else [args.arm]
    entries = []
    for prompt_index in prompt_indices:
        for arm in arms:
            paths = artifact_paths(args.output_root, run_id, arm, prompt_index)
            entries.append(
                {
                    "prompt_index": prompt_index,
                    "arm": arm,
                    "seed": paired_seed(args.base_seed, prompt_index),
                    "prompt": prompts[prompt_index],
                    "video": str(paths["video"]),
                    "manifest": str(paths["manifest"]),
                }
            )
    return entries


def generation_regime(sections):
    regime = dict(REGIME)
    regime.update(
        {
            "sections": sections,
            "num_frames": sections * REGIME["rgb_frames_per_section"],
            "duration_seconds": sections * REGIME["rgb_frames_per_section"] / REGIME["fps"],
        }
    )
    return regime


def tensor_stats(torch, tensor):
    values = tensor.detach().float()
    if values.numel() == 0:
        return {"shape": list(values.shape), "numel": 0}
    finite = torch.isfinite(values)
    result = {
        "shape": list(values.shape),
        "numel": values.numel(),
        "finite": bool(finite.all()),
    }
    if bool(finite.any()):
        valid = values[finite]
        result.update(
            {
                "mean": float(valid.mean().item()),
                "std": float(valid.std(unbiased=False).item()),
                "min": float(valid.min().item()),
                "max": float(valid.max().item()),
                "absmax": float(valid.abs().max().item()),
            }
        )
    return result


def require_finite(torch, tensor, section_index):
    display_index = section_index + 1
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        raise RuntimeError(f"section {display_index} returned an empty or non-tensor latent")
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"non-finite latent detected at section {display_index}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_state():
    status = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).splitlines()
    return {
        "git_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "git_dirty": bool(status),
        "git_status_porcelain": status,
        "driver_path": str(Path(__file__).resolve()),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
    }


def ffmpeg_executable():
    import imageio_ffmpeg

    return Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()


def ffprobe_executable(ffmpeg_path):
    candidates = []
    configured = os.environ.get("FFPROBE_BINARY")
    if configured:
        candidates.append(Path(configured).expanduser())
    path_hit = shutil.which("ffprobe")
    if path_hit:
        candidates.append(Path(path_hit))
    candidates.append(ffmpeg_path.with_name("ffprobe"))
    prefixes = [Path(sys.prefix)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefixes.insert(0, Path(conda_prefix))
    for prefix in prefixes:
        candidates.append(prefix / "bin/ffprobe")
        environments = prefix.parent if prefix.parent.name == "envs" else prefix.parent / "envs"
        candidates.extend(sorted(environments.glob("*/bin/ffprobe")))
        package_cache = prefix.parent.parent / "pkgs"
        candidates.extend(sorted(package_cache.glob("ffmpeg-*/bin/ffprobe"), reverse=True))
    failures = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        try:
            check = subprocess.run(
                [str(candidate), "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{candidate}: {error}")
            continue
        if check.returncode == 0:
            return candidate.resolve()
        failures.append(f"{candidate}: exit {check.returncode}: {check.stderr.strip()}")
    checked = "; ".join(failures) or ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(
        "a runnable ffprobe is required for final MP4 validation; set FFPROBE_BINARY "
        f"to an executable (checked: {checked})"
    )


def parse_rate(value):
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"ffprobe reported invalid frame rate {value!r}") from error
    return float(rate)


def probe_video(video_path, ffprobe_path, expected_frames):
    command = [
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"ffprobe failed with exit code {completed.returncode}: {detail}")
    try:
        streams = json.loads(completed.stdout).get("streams", [])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ffprobe returned invalid JSON: {completed.stdout!r}") from error
    if len(streams) != 1:
        raise RuntimeError(f"ffprobe found {len(streams)} primary video streams; expected 1")
    stream = streams[0]
    codec_name = stream.get("codec_name")
    if codec_name != "h264":
        raise RuntimeError(f"ffprobe reported codec_name={codec_name!r}; expected 'h264' from libx264")
    try:
        frames = int(stream.get("nb_read_frames"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"ffprobe reported invalid nb_read_frames={stream.get('nb_read_frames')!r}") from error
    if frames != expected_frames:
        raise RuntimeError(f"ffprobe counted {frames} frames; expected {expected_frames}")
    fps = parse_rate(stream.get("avg_frame_rate", ""))
    if abs(fps - REGIME["fps"]) > 1e-6:
        raise RuntimeError(f"ffprobe reported fps={fps}; expected {REGIME['fps']}")
    return {
        "codec_name": codec_name,
        "frame_count": frames,
        "fps": fps,
        "avg_frame_rate": stream["avg_frame_rate"],
        "file_size_bytes": video_path.stat().st_size,
    }


def load_pipeline(torch, memory_partial=None):
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
    if memory_partial is not None:
        # Overwrite freshly initialized memory components with Stage A trained
        # weights. The partial also carries patch_long/mid/short convs, but those
        # are backbone components frozen during Stage A (zero movement across
        # checkpoints) and belong to the *training* base, which differs sharply
        # from the distilled base used here (patch_long rel_l2 ~1.17) — loading
        # them would corrupt history patchify. Only the actually trained,
        # base-agnostic memory components are loaded. load_state_dict casts
        # dtype/device per-parameter and consumes no RNG, preserving arm pairing.
        state = torch.load(str(memory_partial), map_location="cpu", weights_only=True)
        skipped = [k for k in state if k.startswith(("patch_long", "patch_mid", "patch_short"))]
        loadable = {
            k: v
            for k, v in state.items()
            if k.startswith("evolving_memory.") or "memory_key_scale" in k
        }
        leftover = set(state) - set(loadable) - set(skipped)
        if leftover:
            raise RuntimeError(f"memory partial has unclassified keys: {sorted(leftover)[:5]}")
        _, unexpected = transformer.load_state_dict(loadable, strict=False)
        if unexpected:
            raise RuntimeError(
                f"memory partial has {len(unexpected)} keys absent from the model: {unexpected[:5]}"
            )
        print(
            f"MEMORY_PARTIAL loaded keys={len(loadable)} skipped_backbone={len(skipped)} "
            f"from {memory_partial}",
            flush=True,
        )
    scheduler = HeliosScheduler.from_pretrained(str(DISTILLED), subfolder="scheduler", stages=3)
    pipe = HeliosPipeline.from_pretrained(
        str(DISTILLED),
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def memory_probes(torch, pipe, write_sigmas, gate_stats):
    memory = pipe.transformer.evolving_memory
    original_update = memory.update

    def tracked_update(evicted_hidden, frame_mask=None, sigma_last=None):
        write_sigmas.append(None if sigma_last is None else float(sigma_last))
        return original_update(evicted_hidden, frame_mask=frame_mask, sigma_last=sigma_last)

    def gate_hook(_module, _inputs, output):
        gate_stats.append(tensor_stats(torch, torch.sigmoid(output)))

    memory.update = tracked_update
    hook = memory.gate_linear.register_forward_hook(gate_hook)
    return memory, original_update, hook


def memory_state_digest(torch, state, write_sigmas, gate_stats):
    if state is None:
        return None
    queue = state["queue"]
    queue_entries = [
        {
            "section_index": int(index),
            "sigma": None if sigma is None else float(sigma),
            "capture": tensor_stats(torch, capture),
        }
        for index, capture, sigma in queue
    ]
    all_sigmas = list(write_sigmas) + [entry["sigma"] for entry in queue_entries]
    return {
        "write_count": len(write_sigmas),
        "write_sigmas": write_sigmas,
        "queue_indices": [entry["section_index"] for entry in queue_entries],
        "queue": queue_entries,
        "per_section_sigmas": all_sigmas,
        "state_tensor": tensor_stats(torch, state["M"]),
        "gate_stats_per_write": gate_stats,
    }


def encode_latents(torch, pipe, latents, video_path, sections, ffmpeg_path, ffprobe_path):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    latent_frames = REGIME["latent_frames_per_section"]
    expected_frames = sections * REGIME["rgb_frames_per_section"]
    vae = pipe.vae
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(vae.device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        vae.device, vae.dtype
    )

    frames_written = 0
    process = None
    temporary = video_path.with_suffix(video_path.suffix + f".tmp.{os.getpid()}.mp4")
    # Own the ffmpeg process so trailer/mux failures propagate through its exit code.
    with tempfile.TemporaryFile() as ffmpeg_stderr:
        try:
            with torch.no_grad():
                for section_index in range(sections):
                    start = section_index * latent_frames
                    chunk = latents[:, :, start : start + latent_frames].to(vae.device, vae.dtype)
                    decoded = vae.decode(chunk / latents_std + latents_mean, return_dict=False)[0]
                    frames = pipe.video_processor.postprocess_video(decoded, output_type="np")[0]
                    for frame in frames:
                        pixels = (frame * 255).round().clip(0, 255).astype("uint8")
                        if pixels.ndim != 3 or pixels.shape[2] != 3:
                            raise RuntimeError(f"decoded frame has invalid shape {pixels.shape}; expected HxWx3")
                        if process is None:
                            height, width = pixels.shape[:2]
                            process = subprocess.Popen(
                                [
                                    str(ffmpeg_path),
                                    "-y",
                                    "-loglevel",
                                    "error",
                                    "-f",
                                    "rawvideo",
                                    "-pixel_format",
                                    "rgb24",
                                    "-video_size",
                                    f"{width}x{height}",
                                    "-framerate",
                                    str(REGIME["fps"]),
                                    "-i",
                                    "-",
                                    "-an",
                                    "-c:v",
                                    "libx264",
                                    "-pix_fmt",
                                    "yuv420p",
                                    str(temporary),
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=ffmpeg_stderr,
                            )
                        process.stdin.write(pixels.tobytes())
                        frames_written += 1
                    del chunk, decoded, frames
            if process is None:
                raise RuntimeError("VAE decode produced no frames")
            process.stdin.close()
            process.stdin = None
            returncode = process.wait()
            ffmpeg_stderr.seek(0)
            diagnostic = ffmpeg_stderr.read().decode(errors="replace").strip()
            if returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed with exit code {returncode}: {diagnostic or 'no diagnostic output'}"
                )
            if frames_written != expected_frames:
                raise RuntimeError(f"decoded {frames_written} frames; expected {expected_frames}")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("ffmpeg exited successfully but produced an empty MP4")
            video_probe = probe_video(temporary, ffprobe_path, expected_frames)
            temporary.replace(video_path)
        except BaseException:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            temporary.unlink(missing_ok=True)
            raise
    return frames_written, video_probe


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args, run_id, prompt_path, prompts):
    if isinstance(prompt_path, list):
        # rep50: one source CSV per prompt index (manifest hashes the exact case file).
        prompt_path = prompt_path[args.prompt_index]
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Torch is intentionally imported only after GPU selection; --dry-run never imports it.
    import torch

    entry = resolved_entries(args, run_id, prompts)[0]
    seed = entry["seed"]
    paths = {key: Path(value) for key, value in entry.items() if key in ("video", "manifest")}
    regime = generation_regime(args.sections)
    memory_partial_info = None
    if args.memory_partial is not None:
        hasher = hashlib.sha256()
        with open(args.memory_partial, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                hasher.update(chunk)
        memory_partial_info = {
            "path": str(args.memory_partial.resolve()),
            "sha256": hasher.hexdigest(),
        }
    digest_payload = {
        "model": str(DISTILLED),
        "memory_module_kwargs": MEMORY_KWARGS,
        "regime": regime,
    }
    # Only fold the partial into the digest when present so that runs without
    # --memory-partial keep the exact r1 config digest.
    if memory_partial_info is not None:
        digest_payload["memory_partial_sha256"] = memory_partial_info["sha256"]
    config_digest = canonical_digest(digest_payload)
    milestones = Milestones()
    timings = {}
    started = time.perf_counter()
    source = source_state()
    ffmpeg_path = ffmpeg_executable()
    ffprobe_path = ffprobe_executable(ffmpeg_path)

    # Global seeding fixes fresh memory-module initialization; the dedicated
    # CUDA generator fixes every pipeline noise draw independently of global RNG.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    load_started = time.perf_counter()
    pipe = load_pipeline(torch, memory_partial=args.memory_partial)
    timings["load_seconds"] = time.perf_counter() - load_started
    if bool(pipe.scheduler.config.use_dynamic_shifting) is not REGIME["use_dynamic_shifting"]:
        raise RuntimeError("Distilled scheduler use_dynamic_shifting does not match the mapped regime")
    if str(pipe.scheduler.config.time_shift_type) != REGIME["time_shift_type"]:
        raise RuntimeError("Distilled scheduler time_shift_type does not match the mapped regime")
    milestones.emit(f"load done arm={args.arm} prompt={args.prompt_index} seed={seed}")

    section = 0
    callback_calls = 0
    steps_per_section = sum(REGIME["stage2_num_inference_steps_list"])
    first_section_steps = steps_per_section * 2

    def section_callback(_pipe, _step_index, _timestep, callback_kwargs):
        nonlocal callback_calls, section
        current = callback_kwargs["latents"]
        require_finite(torch, current, section)
        callback_calls += 1
        completed = callback_calls == first_section_steps or (
            callback_calls > first_section_steps
            and (callback_calls - first_section_steps) % steps_per_section == 0
        )
        if completed:
            milestones.emit(f"section {section + 1}/{args.sections} finite")
            section += 1
        return callback_kwargs

    write_sigmas = []
    gate_stats = []
    memory = original_update = hook = None
    if args.arm == "on":
        memory, original_update, hook = memory_probes(torch, pipe, write_sigmas, gate_stats)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    rollout_started = time.perf_counter()
    try:
        output = pipe(
            prompt=entry["prompt"],
            height=REGIME["height"],
            width=REGIME["width"],
            num_frames=regime["num_frames"],
            output_type="latent",
            guidance_scale=REGIME["guidance_scale"],
            use_dmd=REGIME["use_dmd"],
            is_enable_stage2=REGIME["is_enable_stage2"],
            stage2_num_inference_steps_list=REGIME["stage2_num_inference_steps_list"],
            stage2_num_stages=REGIME["stage2_num_stages"],
            is_amplify_first_chunk=REGIME["is_amplify_first_chunk"],
            use_dynamic_shifting=REGIME["use_dynamic_shifting"],
            time_shift_type=REGIME["time_shift_type"],
            enable_evolving_memory=args.arm == "on",
            generator=generator,
            callback_on_step_end=section_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    finally:
        if hook is not None:
            hook.remove()
        if memory is not None:
            memory.update = original_update
    timings["rollout_seconds"] = time.perf_counter() - rollout_started

    latents = output.frames if hasattr(output, "frames") else output[0]
    require_finite(torch, latents, max(section - 1, 0))
    expected_shape = (
        1,
        16,
        args.sections * REGIME["latent_frames_per_section"],
        REGIME["height"] // 8,
        REGIME["width"] // 8,
    )
    if tuple(latents.shape) != expected_shape:
        raise RuntimeError(f"unexpected latent shape {tuple(latents.shape)}; expected {expected_shape}")
    if section != args.sections:
        raise RuntimeError(f"observed {section} completed sections; expected {args.sections}")

    state_digest = None
    if args.arm == "on":
        state_digest = memory_state_digest(torch, pipe.get_memory_state(), write_sigmas, gate_stats)

    milestones.emit("decode start")
    encode_started = time.perf_counter()
    frames_written, video_probe = encode_latents(
        torch,
        pipe,
        latents,
        paths["video"],
        args.sections,
        ffmpeg_path,
        ffprobe_path,
    )
    timings["decode_encode_seconds"] = time.perf_counter() - encode_started
    milestones.emit(
        f"encode done frames={frames_written} codec=libx264 "
        f"ffprobe_codec={video_probe['codec_name']} path={paths['video']}"
    )

    timings["total_seconds"] = time.perf_counter() - started
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "arm": args.arm,
        "enable_evolving_memory": args.arm == "on",
        "prompt_index": args.prompt_index,
        "prompt": entry["prompt"],
        "prompt_set": args.prompt_set,
        "prompt_source": str(prompt_path),
        "prompt_source_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "base_seed": args.base_seed,
        "seed": seed,
        "rng_parity": (
            "Both arms use the same global seed before model construction and a dedicated CUDA generator "
            "with the same paired seed; evolving-memory read/write operations do not consume that generator."
        ),
        "model": str(DISTILLED),
        "memory_partial": memory_partial_info,
        "source": source,
        "git_commit": source["git_commit"],
        "config_digest_sha256": config_digest,
        "regime": regime,
        "scheduler": {
            "use_dynamic_shifting": bool(pipe.scheduler.config.use_dynamic_shifting),
            "time_shift_type": str(pipe.scheduler.config.time_shift_type),
        },
        "artifacts": {
            "video": str(paths["video"]),
            "video_probe": video_probe,
            "ffmpeg": str(ffmpeg_path),
            "ffprobe": str(ffprobe_path),
        },
        "timings": {key: float(value) for key, value in timings.items()},
        "memory_state": state_digest,
        "latent": tensor_stats(torch, latents),
        "frames_written": frames_written,
        "host": socket.gethostname(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(paths["manifest"], manifest)
    milestones.emit(f"DONE runid={run_id} arm={args.arm} prompt={args.prompt_index} manifest={paths['manifest']}")


def main(argv=None):
    args = parse_args(argv)
    run_id = validate_run_id(args.run_id or utc_run_id())
    if args.prompt_set == "rep50":
        # Per-index case files; run() indexes with the rollout's prompt_index.
        prompt_path, prompts = load_rep50_prompts(args.num_prompts)
    else:
        prompt_path, prompts = load_vs24_prompts(args.num_prompts)
    entries = resolved_entries(args, run_id, prompts)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "run_id": run_id,
                    "prompt_set": args.prompt_set,
                    "prompt_source": (
                        str(prompt_path[0].parent) if isinstance(prompt_path, list) else str(prompt_path)
                    ),
                    "prompt_count": args.num_prompts,
                    "regime": generation_regime(args.sections),
                    "config_digest_sha256": canonical_digest(
                        {
                            "model": str(DISTILLED),
                            "memory_module_kwargs": MEMORY_KWARGS,
                            "regime": generation_regime(args.sections),
                        }
                    ),
                    "entries": entries,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    run(args, run_id, prompt_path, prompts)


if __name__ == "__main__":
    main()

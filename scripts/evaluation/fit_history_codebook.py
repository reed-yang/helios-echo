#!/usr/bin/env python3
"""Fit K-means history-projection codebooks from the Stage-A latent corpus.

The training loader consumes the ``vae_latent`` tensor directly. This tool
therefore identifies whether that tensor is already in Helios model space
before fitting and saves centroids in the same space expected by
``HistoryProjector(mode="codebook")``.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = Path("/mnt/beegfs/siyuan/dataset/human_single_368x640_slice512g/latents")
VAE_CONFIG = Path("/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled/vae/config.json")
LATENT_CHANNELS = 16
HOLDOUT_FRACTION = 0.10
ASSIGNMENT_CHANGE_THRESHOLD = 1e-4
ASSIGNMENT_CHUNK_SIZE = 16384


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, action="append", required=True, help="codebook size; may be repeated")
    parser.add_argument("--num-files", type=int, default=256, help="number of distinct latent files to sample")
    parser.add_argument(
        "--pixels-per-file", type=int, default=2048, help="random C-vectors sampled from each selected file"
    )
    parser.add_argument("--iters", type=int, default=25, help="maximum Lloyd iterations")
    parser.add_argument("--seed", type=int, default=44, help="random seed for file/pixel sampling and initialization")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "output .pt path for one --k; with repeated --k, an output directory "
            "where k<K>.pt files are written"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Stage-A latent directory")
    parser.add_argument("--dry-run", action="store_true", help="report the resolved sampling plan without fitting")
    args = parser.parse_args(argv)

    if any(k < 1 for k in args.k):
        parser.error("each --k must be positive")
    if len(set(args.k)) != len(args.k):
        parser.error("each --k must be distinct")
    if args.num_files < 1 or args.pixels_per_file < 2 or args.iters < 1:
        parser.error("--num-files, --pixels-per-file, and --iters must be positive (pixels >= 2)")
    if len(args.k) > 1 and args.out.suffix:
        parser.error("repeated --k requires --out to be an output directory, not a .pt file")
    return args


def discover_files(dataset_root):
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    files = sorted(path for path in dataset_root.glob("*.pt") if path.is_file())
    if not files:
        raise RuntimeError(f"no .pt files found under {dataset_root}")
    return files


def resolve_output_path(out, k, multiple_k):
    if multiple_k:
        return out / f"k{k}.pt"
    return out


def sha256_file_list(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_vae_constants():
    config = json.loads(VAE_CONFIG.read_text())
    if int(config["z_dim"]) != LATENT_CHANNELS:
        raise ValueError(f"expected z_dim={LATENT_CHANNELS}, got {config['z_dim']} in {VAE_CONFIG}")
    return torch.tensor(config["latents_mean"], dtype=torch.float32), torch.tensor(config["latents_std"], dtype=torch.float32)


def channel_stats(values):
    return values.mean(dim=0), values.std(dim=0, correction=0)


def normalization_score(values):
    """Lower is closer to the expected zero-mean, unit-std model space."""
    mean, std = channel_stats(values)
    return float(mean.abs().mean() + (std - 1.0).abs().mean())


def format_vector(values):
    return "[" + ", ".join(f"{value:.4f}" for value in values.tolist()) + "]"


def print_stats(label, values):
    mean, std = channel_stats(values)
    print(f"{label} mean={format_vector(mean)} std={format_vector(std)} aggregate_std={values.std(correction=0).item():.4f}")


def sample_vectors(selected_files, pixels_per_file, generator):
    """Sample train/held-out C-vectors independently within every selected file."""
    fit_chunks = []
    heldout_chunks = []
    for file_index, path in enumerate(selected_files, start=1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latents = payload.get("vae_latent") if isinstance(payload, dict) else None
        if not torch.is_tensor(latents) or latents.ndim != 5 or latents.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"{path}: expected vae_latent shaped (sections, {LATENT_CHANNELS}, frames, height, width)")
        pixels = latents.float().movedim(1, -1).reshape(-1, LATENT_CHANNELS)
        if pixels.shape[0] < pixels_per_file:
            raise ValueError(f"{path}: only {pixels.shape[0]} pixels, need --pixels-per-file={pixels_per_file}")
        indices = torch.randperm(pixels.shape[0], generator=generator)[:pixels_per_file]
        sampled = pixels[indices]
        heldout_count = max(1, round(pixels_per_file * HOLDOUT_FRACTION))
        fit_chunks.append(sampled[heldout_count:])
        heldout_chunks.append(sampled[:heldout_count])
        if file_index % 50 == 0 or file_index == len(selected_files):
            print(f"SAMPLED files={file_index}/{len(selected_files)}", flush=True)
    return torch.cat(fit_chunks), torch.cat(heldout_chunks)


def choose_model_space(precomputed_values, mean, std):
    """Return model-space data without reapplying the corpus encoder transform.

    The corpus writer applies ``(z_raw - mean) / std`` before saving and the
    Stage-1 loader passes ``vae_latent`` through unchanged. Applying it again
    would therefore put the codebook in a space the pipeline never reads.
    The hypothetical second transform is printed only as a corruption guard.
    """
    second_transform = (precomputed_values - mean) / std
    stored_score = normalization_score(precomputed_values)
    second_transform_score = normalization_score(second_transform)
    aggregate_std = float(precomputed_values.std(correction=0))
    if not 0.60 <= aggregate_std <= 1.30:
        raise RuntimeError(
            "sampled precomputed model-space latents have aggregate std outside the expected range "
            f"(std={aggregate_std:.3f}); refusing to fit"
        )
    print(
        "NORMALIZATION "
        "applied=False source=precomputed_model_space "
        f"stored_score={stored_score:.4f} hypothetical_second_transform_score={second_transform_score:.4f} "
        f"aggregate_std={aggregate_std:.4f}"
    )
    return precomputed_values, False


def assign_nearest(values, centroids):
    assignments = torch.empty(values.shape[0], dtype=torch.long)
    min_distances = torch.empty(values.shape[0], dtype=torch.float32)
    centroid_sq = (centroids * centroids).sum(dim=1)
    for start in range(0, values.shape[0], ASSIGNMENT_CHUNK_SIZE):
        block = values[start : start + ASSIGNMENT_CHUNK_SIZE]
        distances = (block * block).sum(dim=1, keepdim=True) - 2.0 * (block @ centroids.t()) + centroid_sq
        min_distances[start : start + block.shape[0]], assignments[start : start + block.shape[0]] = distances.min(dim=1)
    return assignments, min_distances.clamp_min_(0.0)


def fit_kmeans(values, k, max_iters, generator):
    if k > values.shape[0]:
        raise ValueError(f"k={k} exceeds sampled fit vectors={values.shape[0]}")
    centroids = values[torch.randperm(values.shape[0], generator=generator)[:k]].clone()
    previous_assignments = None
    iters_run = 0
    for iteration in range(1, max_iters + 1):
        assignments, min_distances = assign_nearest(values, centroids)
        inertia = float(min_distances.sum())
        counts = torch.bincount(assignments, minlength=k)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, assignments, values)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
        empty = (~nonempty).nonzero(as_tuple=False).flatten()
        if empty.numel():
            farthest = torch.topk(min_distances, k=empty.numel(), largest=True).indices
            centroids[empty] = values[farthest]
        change_ratio = 1.0 if previous_assignments is None else float((assignments != previous_assignments).float().mean())
        print(
            f"KMEANS k={k} iteration={iteration} inertia={inertia:.6f} "
            f"assignment_change_ratio={change_ratio:.8f} empty_clusters={empty.numel()}",
            flush=True,
        )
        iters_run = iteration
        if previous_assignments is not None and change_ratio < ASSIGNMENT_CHANGE_THRESHOLD and not empty.numel():
            break
        previous_assignments = assignments
    _, final_distances = assign_nearest(values, centroids)
    return centroids.float(), iters_run, float(final_distances.sum())


def heldout_metrics(heldout_values, centroids):
    assignments, distances = assign_nearest(heldout_values, centroids)
    quantized = centroids[assignments]
    mean_l2_error = float(distances.sqrt().mean())
    centered = heldout_values - heldout_values.mean(dim=0, keepdim=True)
    total_variance = float((centered * centered).sum())
    residual_variance = float(((heldout_values - quantized) ** 2).sum())
    variance_retained = 1.0 - residual_variance / total_variance if total_variance > 0.0 else float("nan")
    return mean_l2_error, variance_retained, int(assignments.unique().numel())


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main(argv=None):
    args = parse_args(argv)
    files = discover_files(args.dataset_root)
    if args.num_files > len(files):
        raise ValueError(f"--num-files={args.num_files} exceeds resolved corpus size {len(files)}")
    sampling_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    selected_indices = torch.randperm(len(files), generator=sampling_generator)[: args.num_files].tolist()
    selected_files = [files[index] for index in selected_indices]
    selected_file_hash = sha256_file_list(selected_files)
    fit_per_file = args.pixels_per_file - max(1, round(args.pixels_per_file * HOLDOUT_FRACTION))
    print(
        f"PLAN dataset_root={args.dataset_root} corpus_files={len(files)} selected_files={len(selected_files)} "
        f"fit_pixels_per_file={fit_per_file} heldout_pixels_per_file={args.pixels_per_file - fit_per_file} "
        f"fit_pixels={fit_per_file * len(selected_files)} seed={args.seed} file_list_sha256={selected_file_hash}",
        flush=True,
    )
    if args.dry_run:
        return

    raw_fit, raw_heldout = sample_vectors(selected_files, args.pixels_per_file, sampling_generator)
    mean, std = load_vae_constants()
    model_fit, normalization_applied = choose_model_space(raw_fit, mean, std)
    model_heldout = raw_heldout
    print_stats("SAMPLED_MODEL_SPACE", model_fit)

    for k in args.k:
        k_generator = torch.Generator(device="cpu").manual_seed(args.seed + k)
        centroids, iters_run, final_inertia = fit_kmeans(model_fit, k, args.iters, k_generator)
        print_stats("CENTROIDS", centroids)
        mean_l2_error, variance_retained, used_centroids = heldout_metrics(model_heldout, centroids)
        print(
            f"ROUNDTRIP k={k} heldout_pixels={model_heldout.shape[0]} mean_l2_error={mean_l2_error:.6f} "
            f"variance_retained={variance_retained:.6f}"
        )
        print(f"CENTROID_USAGE k={k} used={used_centroids}/{k}")

        out_path = resolve_output_path(args.out, k, len(args.k) > 1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "k": k,
            "num_files": len(selected_files),
            "pixels_per_file": args.pixels_per_file,
            "total_pixels": int(model_fit.shape[0]),
            "heldout_pixels": int(model_heldout.shape[0]),
            "iters_run": iters_run,
            "final_inertia": final_inertia,
            "seed": args.seed,
            "dataset_root": str(args.dataset_root),
            "normalization_applied": normalization_applied,
            "normalization_constants": {
                "latents_mean": mean.tolist(),
                "latents_std": std.tolist(),
                "formula": "(z_raw - latents_mean) / latents_std" if normalization_applied else "none (input already model-space)",
            },
            "sampled_file_list_sha256": selected_file_hash,
            "script_git_commit": git_commit(),
        }
        torch.save({"codebook": centroids, "meta": metadata}, out_path)
        print(f"CODEBOOK k={k} pixels={model_fit.shape[0]} inertia={final_inertia:.6f} out={out_path}")
    print("EXIT_CODE=0")


if __name__ == "__main__":
    main()

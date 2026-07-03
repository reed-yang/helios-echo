# Helios on Wan2.2-TI2V-5B (Stage-1 init + post)

Port of the Helios autoregressive video-DiT from the **Wan2.1-T2V-14B** base to the
**Wan2.2-TI2V-5B** base, replicating the Stage-1 `init` + `post` recipe, with eval.

## Why / what changed

Wan2.2-TI2V-5B is a *dense* 5B model with a higher-compression VAE. The deltas that matter:

| | Wan2.1-T2V-14B | Wan2.2-TI2V-5B |
|---|---|---|
| VAE latent channels (z_dim) | 16 | **48** |
| VAE temporal compression | 4× | 4× (unchanged) |
| VAE spatial compression | 8× | **16×** |
| transformer inner_dim | 5120 (40×128) | **3072 (24×128)** |
| num_layers | 40 | **30** |
| attention_head_dim | 128 | 128 (unchanged → `rope_dim=(44,42,42)` still valid) |
| ffn_dim | 13824 | **14336** |
| patch_size / text_dim | (1,2,2) / 4096 | unchanged |

Temporal compression is still 4×, so **chunk geometry is unchanged**: `latent_window_size=9`
→ `(9-1)*4+1 = 33` RGB frames/chunk.

Because the transformer dims load automatically from the Wan2.2 `config.json` via
`HeliosTransformer3DModel.from_pretrained`, the **only required code change** was one Wan2.1-coupled
hardcode:

- `helios/modules/transformer_helios.py` `initialize_weight_from_another_conv3d`: `weight[:, :16]`
  → `weight[:, : self.patch_short.in_channels]` (was assuming 16 latent channels; now 48).

Two more changes are about resolution, not the model:

- **Resolution 384×640, not 368×640.** At 16× spatial, `368/16 = 23` is **odd** → the stride-2
  `patch_embedding` drops a row and breaks unpatchify. `384/16 = 24` is even (and `÷8` for
  `patch_long`). 384×640 is an existing bucket and visually the same low-res class.
- `helios/dataset/dataloader_mp4_dist.py`: added `resolution_bucket_options[642]` = the `640` set
  **minus** `(368,640)`, so ~16:9 source snaps to the even-latent `(384,640)`. Encode with
  `--resolution 642`. The `640` set (Wan2.1 / cfr368 pipeline) is untouched.

`_cp_plan` still hardcodes `range(40)`; harmless for 30 layers (superset), only relevant to
`--enable_parallelism`. The diffusers/transformers in the `helios` env already support Wan2.2
(`AutoencoderKLWan`, `WanTransformer3DModel`) — no upgrade needed.

## Files

- Weights: `/mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers` (transformer/vae/text_encoder/tokenizer).
- `scripts/wan22/verify_load.py` — standalone load check (all checks PASS: 5.04B, missing=6 = the patch_*).
- `scripts/wan22/make_subset_json.py` — carve PoC/smoke manifests from `offload_cfr_int.json`.
- `scripts/wan22/encode_wan22.sbatch` — encode latents with the Wan2.2 VAE (node8, 6 GPU, `--resolution 642`).
- `scripts/training/configs/stage1_init_wan22{,_smoke}.yaml`, `stage1_post_wan22.yaml`.
- `scripts/training/train_stage1_wan22_node8.sbatch` — single-node train (node8, 6 usable GPU).
- `scripts/wan22/merge_lora_wan22.py` — merge LoRA + multi-term patch into a standalone transformer.
- `scripts/wan22/infer_wan22_node7.sbatch` — t2v eval inference (node7).

## Cluster notes

- **node8 (c-node08) = training/data-prep.** GPU **1 is in `Prohibited` compute mode** (kept hot for
  inference). Slurm allocates physical 0–6 (incl. #1), so all node8 jobs set
  `CUDA_VISIBLE_DEVICES=0,2,3,4,5,6` and run **6 ranks** (`--ntasks-per-node=6 --gres=gpu:h200:7`).
- **node7 (c-node07) = inference/eval.**
- env: `/mnt/beegfs/yuheng/miniconda3/envs/helios` (torch 2.10, diffusers 0.39.0.dev0).

## Runbook

```bash
DATA=/mnt/beegfs/dataset/video_single_24FPS
PY=/mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python

# 0. (one-time) verify the port loads the Wan2.2 base
PYTHONPATH=$PWD $PY scripts/wan22/verify_load.py

# 1. subset manifests (filter >=121 frames, take N)
$PY scripts/wan22/make_subset_json.py --src $DATA/offload_cfr_int.json \
    --out $DATA/offload_cfr_int_wan22_poc.json --n 25000

# 2. encode latents with the Wan2.2 VAE (384x640, z_dim=48) -> idempotent, resubmittable
JSON=$DATA/offload_cfr_int_wan22_poc.json OUT=$DATA/latents_wan22_384x640_poc BATCH_SIZE=4 \
    sbatch scripts/wan22/encode_wan22.sbatch

# 3. Stage-1 INIT (from raw Wan2.2; lr 5e-5 constant; gb 6x4x4=96)
CONFIG=scripts/training/configs/stage1_init_wan22.yaml sbatch scripts/training/train_stage1_wan22_node8.sbatch

# 4. merge init LoRA + multi-term patch -> standalone transformer
$PY scripts/wan22/merge_lora_wan22.py \
    --base /mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers \
    --lora    /mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22/checkpoint-5500/pytorch_lora_weights.safetensors \
    --partial /mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22/checkpoint-5500/transformer_partial.pth \
    --out     /mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22_merged

# 5. Stage-1 POST (from merged init; lr 3e-5, 500 warmup) -> then merge again
CONFIG=scripts/training/configs/stage1_post_wan22.yaml sbatch scripts/training/train_stage1_wan22_node8.sbatch

# 6. eval: t2v inference on node7, then run the existing metric suite on the output folder
TRANSFORMER=/mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22_merged/transformer \
    LABEL=init5500 sbatch scripts/wan22/infer_wan22_node7.sbatch
```

## Status (validated)

- Port loads cleanly (verify_load: ALL PASS, 5.04B, only the 6 multi-term-patch keys freshly init).
- Smoke encode → 48-ch latents `(num_chunks,48,9,24,40)`, normalized, UMT5 embeds.
- Smoke train (30 steps, 6 GPU): finite flow loss ~0.15–0.45, LoRA+patch trainable, checkpoint saved,
  validation video generated → **train + inference paths both work**.

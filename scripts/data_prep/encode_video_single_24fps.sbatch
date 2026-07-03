#!/bin/bash
#SBATCH --job-name=helios_enc24
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8          # one task per GPU -> 32 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=c-node04,c-node07,mc-node01,mc-node02   # 4 fast idle nodes (skip slow c-node08)
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/enc24_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/enc24_%j.out
# Encode the FULL video_single_24FPS train set (~1.2M clips) -> 384x640 latents (.pt with
# prompt_embed_short + prompt_embed_long) on 32 GPUs. Idempotent: skips clips whose .pt exists,
# so a timed-out / killed job can simply be resubmitted to continue. One srun task per GPU +
# env rendezvous (NO nested torch.distributed.run, which fails to bind c10d across nodes).
# Captions come from caption_versions.json (built by build_caption_versions.py from cap=[short,long]).
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
DATA=/mnt/beegfs/dataset/video_single_24FPS
JSON=${JSON:-$DATA/train_24fps.json}
VIDEO_DIR=${VIDEO_DIR:-$DATA/video_24fps}
OUT=${OUT:-$DATA/latents_short_384x640}
REWRITTEN=${REWRITTEN:-$DATA/caption_versions.json}
BATCH_SIZE=${BATCH_SIZE:-2}
# Encoder components (tokenizer/vae/text_encoder) from a PLAIN LOCAL DIR (not an HF repo id):
# 32 concurrent ranks racing on HF-Hub snapshot resolution intermittently report missing shards.
ENCODER_SRC=/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled
mkdir -p "$OUT"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29533
export WORLD_SIZE=$SLURM_NTASKS
echo "encode24: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT nodelist=$SLURM_JOB_NODELIST out=$OUT json=$JSON"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_enc_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_enc_${SLURM_JOB_ID}_${SLURM_NODEID}
  # node-local TMPDIR: Triton gcc-builds its launcher .c here; beegfs TMPDIR + many ranks -> ENOENT race.
  export TMPDIR=/tmp/tmp_'"$USER"'_enc_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  # Serialize gcc per node (beegfs concurrent conda-header reads transiently ENOENT -> gcc exit 1
  # -> kill-on-bad-exit kills the job). Triton _build honors $CC.
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python tools/offload_data/get_short-latents.py \
    --pretrained_model_name_or_path '"$ENCODER_SRC"' \
    --json_file '"$JSON"' \
    --video_folder '"$VIDEO_DIR"' \
    --output_latent_folder '"$OUT"' \
    --rewritten_json '"$REWRITTEN"' \
    --caption_versions short,long \
    --batch_size '"$BATCH_SIZE"'
'
echo "ENCODE24 DONE $(date +%s); pt files: $(ls ${OUT}/*.pt 2>/dev/null | wc -l)"

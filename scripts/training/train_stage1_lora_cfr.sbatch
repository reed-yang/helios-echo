#!/bin/bash
#SBATCH --job-name=helios_loracfr
#SBATCH --nodes=2
#SBATCH --nodelist=c-node05,c-node06   # 2 good nodes = 16 GPU (LoRA run); avoid c-node07 (bad NCCL)
#SBATCH --ntasks-per-node=8            # one task per GPU -> 16 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_loracfr_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_loracfr_%j.out
# LoRA r128 fine-tune on the cfr_int / captions_30b_32f corpus (same data + global batch as the
# full-FT run; lr 3e-5 / r128 / alpha128 reuse the prior stage1_video_single_24fps LoRA recipe).
# GLOBAL BATCH 32 = 2 nodes x 8 GPU x bs2 x accum1. Env-based 1-task-per-GPU DDP launcher (NO
# DeepSpeed needed for LoRA) + node-local TMPDIR/cache + gcc serialization. Idempotent restart:
# resume_from_checkpoint=latest (DCP ckpts world-size locked to 16 -> resume on 16 GPU).
# NOTE: c-node07 has a history of deadlocking the first multi-node NCCL collective; if this job hangs
# before step 1, swap c-node07 -> c-node06 in --nodelist and resubmit.
# Launch:  sbatch scripts/training/train_stage1_lora_cfr.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_lora_cfr.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29507
export WORLD_SIZE=$SLURM_NTASKS
echo "train_loracfr: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=online
  export WANDB_API_KEY=$(cat /mnt/beegfs/xiangbo/.wandb_key 2>/dev/null)
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_lora_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_lora_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_lora_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_LORACFR DONE $(date +%s)"

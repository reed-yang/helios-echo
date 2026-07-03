#!/bin/bash
#SBATCH --job-name=helios_reprowan
#SBATCH --nodes=2
#SBATCH --nodelist=c-node05,c-node06   # c07 hangs multi-node NCCL -> dropped; used for inference instead
#SBATCH --ntasks-per-node=8            # one task per GPU -> 16 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_reprowan_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_reprowan_%j.out
# REPRODUCE Helios Stage 1 from the RAW Wan2.1-T2V-14B-Diffusers base (not continue-training): LoRA r128,
# lr 5e-5, multi-term-memory patches + patch_embedding trained from random init, on our cfr_int latents,
# 2 epochs. GLOBAL BATCH 48 = 3 nodes x 8 GPU x bs2 x accum1. Env-based 1-task-per-GPU DDP launcher (NO
# DeepSpeed) + node-local TMPDIR/cache + gcc serialization. resume_from_checkpoint=latest (DCP ckpts
# world-size locked to 24 -> resume on the same 24-GPU set).
# WARNING: c-node07 has a history of deadlocking the FIRST multi-node NCCL collective. The user mandated
# c05/06/07 with NO fallback -> if this hangs before step 1 (NCCL timeout / no progress), STOP and report;
# do NOT auto-substitute a node.
# Launch:  sbatch scripts/training/train_stage1_repro_wan.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_repro_wan.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29508
export WORLD_SIZE=$SLURM_NTASKS
echo "train_reprowan: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

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
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_reprowan_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_reprowan_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_reprowan_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_REPROWAN DONE $(date +%s)"

#!/bin/bash
#SBATCH --job-name=helios_vs24
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8          # one task per GPU -> 32 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c-node08          # c-node08 GPU1 runs hot/slow -> reserved for the inference/eval node
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_vs24_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_vs24_%j.out
# Full video_single_24FPS single-prompt continue-training: LoRA r128, GLOBAL BATCH 64
# (= 4 nodes x 8 GPU x bs2 x accum1), ~18.8k steps ~= 1 epoch over ~1.2M clips.
# Env-based 1-task-per-GPU launcher + node-local TMPDIR/cache + gcc serialization
# (defeats the beegfs concurrent-header-read race). Model components load from LOCAL DIRS
# (config) to avoid the multi-rank HF-Hub snapshot race. c-node07 is INCLUDED (only c-node08
# is excluded); if it hangs the first NCCL collective, exclude it and resubmit.
# Launch:  sbatch scripts/training/train_stage1_video_single_24fps.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_video_single_24fps.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29505
export WORLD_SIZE=$SLURM_NTASKS
echo "train_vs24: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=offline
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_VS24 DONE $(date +%s)"

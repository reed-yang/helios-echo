#!/bin/bash
#SBATCH --job-name=helios_lora368
#SBATCH --nodes=8
#SBATCH --nodelist=mc-node01,mc-node02,c-node03,c-node04,c-node05,c-node06,c-node07,c-node08
#SBATCH --ntasks-per-node=8            # one task per GPU -> 64 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_lora368_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_lora368_%j.out
# Continue-train the stage1_lora_cfr LoRA from checkpoint-9000 on the merged ~388k-clip corpus at 368x640.
# GLOBAL BATCH 128 = 8 nodes x 8 GPU x bs2 x accum1. Env-based 1-task-per-GPU DDP launcher (NO DeepSpeed).
# output_dir is SEEDED below with a symlink checkpoint-9000 -> .../stage1_lora_cfr/checkpoint-9000, so the
# config's resume_from_checkpoint=latest loads 9000 on the first launch and 9500+ on any crash-resume
# (skip_dataloader_dcp=true -> resume is world-size agnostic; only random_states_0.pkl exists, tolerated).
#
# WARNING c-node07 has a history of deadlocking the first multi-node NCCL collective. WATCH step 0.
# 6-NODE FALLBACK (if c07 deadlocks): scancel, set max_train_steps:17000 in the yaml (gb96 -> ~2 epochs), then
#   sbatch --nodes=6 --nodelist=mc-node01,mc-node02,c-node03,c-node04,c-node05,c-node06 \
#          scripts/training/train_stage1_lora_cfr_368.sbatch
# Launch:  sbatch scripts/training/train_stage1_lora_cfr_368.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_lora_cfr_368.yaml}

# Seed the new output_dir with a symlink to checkpoint-9000 so resume_from_checkpoint=latest finds it.
OUTDIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368
SRC9000=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr/checkpoint-9000
mkdir -p "$OUTDIR"
if [ ! -e "$OUTDIR/checkpoint-9000" ]; then
  ln -sfn "$SRC9000" "$OUTDIR/checkpoint-9000"
  echo "seeded $OUTDIR/checkpoint-9000 -> $SRC9000"
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29515
export WORLD_SIZE=$SLURM_NTASKS
echo "train_lora368: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

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
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_lora368_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_lora368_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_lora368_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_LORA368 DONE $(date +%s)"

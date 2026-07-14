#!/bin/bash
#SBATCH --job-name=helios_smoke_neworg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2            # 2 GPUs is plenty for a 3-step data-path smoke
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/smoke_neworg_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/smoke_neworg_%j.out
# SMOKE TEST: continue-train the stage1 LoRA from checkpoint-27500 for 3 steps on the NEW
# canonical tree (/mnt/beegfs/dataset/helios_organized/{animal,ikea_asm,music_performance}/latents)
# to confirm the reorganized data loads + trains with no bug. Writes to a THROWAWAY output_dir
# (smoke_neworg_368) seeded with a symlink to the real checkpoint-27500 (loaded read-only).
# Pin a node with free GPUs via:  sbatch --nodelist=c-node07 scripts/training/smoke_neworg.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/smoke_neworg_368.yaml}

# Seed output_dir/checkpoint-27500 -> real checkpoint so resume_from_checkpoint=latest loads it.
OUTDIR=/mnt/beegfs/xiangbo/helios_runs/smoke_neworg_368
SRC27500=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_reweight_368/checkpoint-27500
mkdir -p "$OUTDIR"
if [ ! -e "$OUTDIR/checkpoint-27500" ]; then
  ln -sfn "$SRC27500" "$OUTDIR/checkpoint-27500"
  echo "seeded $OUTDIR/checkpoint-27500 -> $SRC27500"
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29533
export WORLD_SIZE=$SLURM_NTASKS
echo "smoke_neworg: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

srun --mpi=none --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=offline
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export HELIOS_FORCE_LR=1
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_smoke_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_smoke_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_smoke_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "SMOKE_NEWORG DONE $(date +%s)"

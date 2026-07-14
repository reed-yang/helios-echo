#!/bin/bash
#SBATCH --job-name=helios_neworg
#SBATCH --nodes=1
#SBATCH --nodelist=c-node05
#SBATCH --ntasks-per-node=8            # one task per GPU -> 8 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_neworg_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_neworg_%j.out
# Continue-train the stage1 LoRA from stage1_lora_reweight_368/checkpoint-27500 for 1000 more steps
# (27500 -> 28500) on the NEW canonical tree /mnt/beegfs/dataset/helios_organized (all 12 sources at
# natural count). SINGLE NODE c-node05: 8 GPU x bs3 x accum1 = global batch 24. checkpointing_steps=500
# -> saves checkpoint-28000 and checkpoint-28500. output_dir is SEEDED below with a symlink
# checkpoint-27500 -> stage1_lora_reweight_368/checkpoint-27500 so resume_from_checkpoint=latest loads
# it (skip_dataloader_dcp=true -> resume is world-size agnostic, 40->8 ranks). HELIOS_FORCE_LR=1
# re-applies config lr 1e-5 over the restored optimizer state.
# Launch:  sbatch scripts/training/train_stage1_lora_neworg.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_lora_neworg_368.yaml}

# Seed the new output_dir with a symlink to checkpoint-27500 so resume_from_checkpoint=latest finds it.
OUTDIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_neworg_368
SRC27500=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_reweight_368/checkpoint-27500
mkdir -p "$OUTDIR"
if [ ! -e "$OUTDIR/checkpoint-27500" ]; then
  ln -sfn "$SRC27500" "$OUTDIR/checkpoint-27500"
  echo "seeded $OUTDIR/checkpoint-27500 -> $SRC27500"
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29537
export WORLD_SIZE=$SLURM_NTASKS
echo "train_neworg: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

srun --mpi=none --kill-on-bad-exit=1 bash -c '
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
  export HELIOS_FORCE_LR=1
  export HELIOS_THROTTLE_FREE=1
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_neworg_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_neworg_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_neworg_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_NEWORG DONE $(date +%s)"

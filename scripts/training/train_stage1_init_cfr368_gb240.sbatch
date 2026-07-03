#!/bin/bash
#SBATCH --job-name=helios_i240
#SBATCH --nodes=5
#SBATCH --nodelist=mc-node01,c-node03,c-node04,c-node05,c-node06
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_i240_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_i240_%j.out
# Stage-1-INIT (from raw Wan2.1-T2V-14B) on the FULL ~432k-clip 368x640 corpus (batches 1-3),
# hyperparameter-aligned (lr 5e-5 constant, gb128 = 4 nodes x 8 GPU x bs4, 5500 steps) AND speed-optimized:
#   (dropped c-node05/06 -> nodes 1-4; c-node06 killed the run twice. gb128 = the recipe's exact global batch.)
#   HELIOS_THROTTLE_FREE=1   -> skip per-step free_memory() (gc.collect+empty_cache): -26%, identical training
#   (find_unused stays ON: find_unused=OFF+touch tested SLOWER at multi-node)
# Both are mathematically identical to stock training. See docs/STAGE1_SPEED_OPTIMIZATION.md.
# Launch:  sbatch scripts/training/train_stage1_init_cfr368.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_init_cfr368_gb240.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29533
export WORLD_SIZE=$SLURM_NTASKS
echo "train_init368: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodes=$SLURM_JOB_NODELIST"

srun --mpi=none --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=online
  export WANDB_API_KEY=$(cat /mnt/beegfs/xiangbo/.wandb_key 2>/dev/null)
  export TOKENIZERS_PARALLELISM=false
  # ---- speed knobs (accuracy-neutral) ----
  export HELIOS_THROTTLE_FREE=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_init368_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_init368_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_init368_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_INIT368 DONE $(date +%s)"

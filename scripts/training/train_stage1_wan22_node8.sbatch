#!/bin/bash
#SBATCH --job-name=helios_wan22
#SBATCH --nodes=1
#SBATCH --nodelist=c-node08          # TRAINING node (user-assigned). GPU1 Prohibited -> 6 usable H200.
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:h200:7
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_wan22_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_wan22_%j.out
# Stage-1 (init or post) for the Wan2.2-TI2V-5B base on a SINGLE node (c-node08, 7 GPU).
#   CONFIG=scripts/training/configs/stage1_init_wan22_smoke.yaml  sbatch ... (smoke)
#   CONFIG=scripts/training/configs/stage1_init_wan22.yaml        sbatch ... (init,  gb112, 5500)
#   CONFIG=scripts/training/configs/stage1_post_wan22.yaml        sbatch ... (post,  gb112, 7500)
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_init_wan22.yaml}
WANDB_MODE=${WANDB_MODE:-online}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29533
export WORLD_SIZE=$SLURM_NTASKS
echo "train_wan22: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE='"$WANDB_MODE"'
  export WANDB_API_KEY=$(cat /mnt/beegfs/xiangbo/.wandb_key 2>/dev/null || true)
  export TOKENIZERS_PARALLELISM=false
  # node8: physical GPU1 is in Prohibited compute mode (kept hot for inference). Slurm allocates
  # physical 0-6 (incl. #1), so restrict to the 6 Default GPUs -> set_device(LOCAL_RANK 0-5) is safe.
  export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6
  # ---- speed knobs (accuracy-neutral; see docs/STAGE1_SPEED_OPTIMIZATION.md) ----
  export HELIOS_THROTTLE_FREE=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_wan22_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_wan22_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_wan22_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_WAN22 DONE $(date +%s)"

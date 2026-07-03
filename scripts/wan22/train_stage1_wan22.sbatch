#!/bin/bash
#SBATCH --job-name=helios_wan22tr
#SBATCH --nodes=1
#SBATCH --nodelist=c-node07          # TRAINING node (user-assigned). node8 is inference-only.
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_wan22full_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_wan22full_%j.out
# Full-data Stage-1 (faithful Helios recipe) on a SINGLE node. Node-agnostic: dynamic GPU selection
# drops any Prohibited GPU (so it also works on node8's 6 GPU if ever needed).
#   CONFIG=scripts/training/configs/stage1_init_wan22full.yaml sbatch scripts/wan22/train_stage1_wan22.sbatch
#   CONFIG=scripts/training/configs/stage1_post_wan22full.yaml sbatch scripts/wan22/train_stage1_wan22.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_init_wan22full.yaml}
WANDB_MODE=${WANDB_MODE:-online}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29535
export WORLD_SIZE=$SLURM_NTASKS
echo "train_wan22full: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG node=$SLURM_JOB_NODELIST"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE='"$WANDB_MODE"'
  export WANDB_API_KEY=$(cat /mnt/beegfs/xiangbo/.wandb_key 2>/dev/null || true)
  export TOKENIZERS_PARALLELISM=false
  # Node-agnostic GPU selection: keep Slurm-allocated GPUs, drop any Prohibited (node8 GPU1).
  PROHIB=$(nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader | awk -F", " "\$2!=\"Default\"{print \$1}")
  NEW=""; for g in ${CUDA_VISIBLE_DEVICES//,/ }; do keep=1; for p in $PROHIB; do [ "$g" = "$p" ] && keep=0; done; [ $keep -eq 1 ] && NEW="$NEW,$g"; done
  export CUDA_VISIBLE_DEVICES="${NEW#,}"
  export HELIOS_THROTTLE_FREE=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_wan22full_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_wan22full_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_wan22full_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_WAN22FULL DONE $(date +%s)"

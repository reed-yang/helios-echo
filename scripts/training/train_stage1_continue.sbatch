#!/bin/bash
#SBATCH --job-name=helios_s1_continue
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8          # one task per GPU -> 32 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c-node08          # node8 GPU1 runs hot -> keep it for inference, not training
#SBATCH --time=12:00:00             # ~4h needed for 1050 steps; generous margin (default cap was 1h -> killed at step 200)
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_%j.out
# 4 nodes x 8 H200 = 32 GPUs. Stage-1 continue-training of Helios from Helios-Base.
# Uses one srun task per GPU + env-based rendezvous (the per-node nested-launcher
# rendezvous fails to bind its c10d store across nodes here: "server socket has failed
# to listen"). accelerate inside train_helios.py reads RANK/WORLD_SIZE/LOCAL_RANK/
# MASTER_ADDR/MASTER_PORT. Launch:  sbatch scripts/training/train_stage1_continue.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=scripts/training/configs/stage1_continue_ohv.yaml

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29501
export WORLD_SIZE=$SLURM_NTASKS
echo "train: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT nodelist=$SLURM_JOB_NODELIST"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  # NOTE: do NOT set HF_HUB_OFFLINE=1 — Helios fetches the flash-attn3 kernel from
  # kernels-community at runtime (now pre-cached). The 32-rank HF snapshot race is avoided
  # by loading model components from LOCAL DIRS (see config pretrained_model_name_or_path).
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=offline
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  # env-based rendezvous that accelerate/torch read:
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  # node-local Triton/inductor cache (beegfs cache races across ranks -> missing .cubin):
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_${SLURM_JOB_ID}_${SLURM_NODEID}
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN DONE $(date +%s)"

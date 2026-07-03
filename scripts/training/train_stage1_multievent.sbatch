#!/bin/bash
#SBATCH --job-name=helios_s1_multievent
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8          # one task per GPU -> 32 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c-node08          # node8 GPU1 runs hot -> keep it for inference, not training
#SBATCH --time=12:00:00             # ~4h for 1050 steps; generous margin (default cap was 1h)
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_me_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_me_%j.out
# 4 nodes x 8 H200 = 32 GPUs. Stage-1 *multi-event* (prompt-switching) continue-training of
# Helios from Helios-Base. Same env-based 1-task-per-GPU launcher as train_stage1_continue.sbatch
# (nested per-node launchers fail to bind the c10d store across nodes here). Multi-event behavior
# is entirely data-driven: the .pt latents carry per-event prompt embeds + a chunk->event map.
# Launch:  sbatch scripts/training/train_stage1_multievent.sbatch
# Override CONFIG=...stage1_multievent_ohv_smoke.yaml for the smoke run.
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_multievent_ohv.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29502
export WORLD_SIZE=$SLURM_NTASKS
echo "train_me: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  # do NOT set HF_HUB_OFFLINE=1 (flash-attn3 kernel is fetched at runtime; pre-cached).
  # The 32-rank HF snapshot race is avoided by loading model components from LOCAL DIRS.
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=offline
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_${SLURM_JOB_ID}_${SLURM_NODEID}
  # NODE-LOCAL TMPDIR: Triton JIT-builds its launcher .c via gcc in $TMPDIR. The inherited
  # beegfs TMPDIR (/mnt/beegfs/.../claude-*) + 32 concurrent compiles caused transient beegfs
  # ENOENT on Python headers ("fatal error: pystate.h: No such file") -> gcc exit 1 -> job dies.
  export TMPDIR=/tmp/tmp_'"$USER"'_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_ME DONE $(date +%s)"

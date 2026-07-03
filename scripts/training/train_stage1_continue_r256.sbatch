#!/bin/bash
#SBATCH --job-name=helios_s1_r256
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8          # one task per GPU -> 32 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c-node08          # node8 GPU1 runs hot -> keep it for inference, not training
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_r256_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_r256_%j.out
# Single-prompt continue-training rerun: LoRA r256, GLOBAL BATCH 32 (= 4 nodes x 8 GPU x bs1 x accum1).
# Same env-based 1-task-per-GPU launcher + node-local TMPDIR/cache as the multi-event run
# (avoids the cross-node c10d bind failure AND the beegfs-TMPDIR Triton gcc race). Model components
# load from LOCAL DIRS (config) to avoid the multi-rank HF-Hub snapshot race.
# Launch:  sbatch scripts/training/train_stage1_continue_r256.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_continue_ohv_r256.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29503
export WORLD_SIZE=$SLURM_NTASKS
echo "train_r256: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

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
  # node-local TMPDIR: Triton gcc-builds its launcher .c here; beegfs TMPDIR + many ranks -> ENOENT race.
  export TMPDIR=/tmp/tmp_'"$USER"'_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  # Serialize gcc per node so the 8 ranks/node do NOT JIT-compile Triton/cuda_utils.c
  # simultaneously: concurrent gcc reads of the beegfs conda Python headers transiently fail
  # ("fatal error: cpython/*.h: No such file") -> gcc exit 1 -> kill-on-bad-exit kills the job.
  # Triton _build honors $CC; the wrapper flock-serializes + retries gcc. (Stagger alone failed:
  # ranks resync at the DDP barrier and hit the first-forward compile together anyway.)
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_R256 DONE $(date +%s)"

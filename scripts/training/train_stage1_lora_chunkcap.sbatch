#!/bin/bash
#SBATCH --job-name=helios_chunkcap
#SBATCH --nodes=5
#SBATCH --nodelist=mc-node01,c-node03,c-node04,c-node05,c-node06
#SBATCH --ntasks-per-node=8            # one task per GPU -> 40 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_chunkcap_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_chunkcap_%j.out
# Chunk-aligned caption SFT (docs/CHUNK_ALIGNED_CAPTIONING.md): continue the lora368-correct
# LoRA from checkpoint-21000 on latents_chunkcap_368x640 — every 33-frame chunk trains against
# its own Gemini caption (event_idx_per_chunk), teaching per-chunk prompt binding.
# Same launcher pattern as train_stage1_lora_cfr_368_correct.sbatch (HELIOS_FORCE_LR keeps
# lr 1e-5 on resume; HELIOS_THROTTLE_FREE=1 validated -26% step time).
# Launch:  sbatch scripts/training/train_stage1_lora_chunkcap.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_lora_chunkcap_368.yaml}

# Seed output_dir with a symlink to checkpoint-21000 so resume_from_checkpoint=latest finds it
# on first launch (21500+ on crash-resume). Link ONLY the plain checkpoint dir.
OUTDIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_chunkcap_368
SRC21000=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368_correct/checkpoint-21000
mkdir -p "$OUTDIR"
if [ ! -e "$OUTDIR/checkpoint-21000" ]; then
  ln -sfn "$SRC21000" "$OUTDIR/checkpoint-21000"
  echo "seeded $OUTDIR/checkpoint-21000 -> $SRC21000"
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29518
export WORLD_SIZE=$SLURM_NTASKS
echo "train_chunkcap: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

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
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_chunkcap_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_chunkcap_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_chunkcap_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_CHUNKCAP DONE $(date +%s)"

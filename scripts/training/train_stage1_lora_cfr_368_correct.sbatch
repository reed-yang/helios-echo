#!/bin/bash
#SBATCH --job-name=helios_lora368c
#SBATCH --nodes=5
#SBATCH --nodelist=mc-node01,c-node03,c-node04,c-node05,c-node06
#SBATCH --ntasks-per-node=8            # one task per GPU -> 40 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_lora368c_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_lora368c_%j.out
# Continue-train the stage1_lora_cfr_368 LoRA from checkpoint-17000 with the full author-recommended
# correct.yaml recipe (docs/HELIOS_CORRECT_RECIPE.md): Blur (corrupt random), I2V drop 0.1, Saturation.
# GLOBAL BATCH 160 = 5 nodes x 8 GPU x bs2 x accum2. LR 3e-5 -> 1e-5 via HELIOS_FORCE_LR=1 (load_state
# would otherwise restore the checkpoint's base_lrs and silently keep 3e-5).
# output_dir is SEEDED below with a symlink checkpoint-17000 -> .../stage1_lora_cfr_368/checkpoint-17000,
# so resume_from_checkpoint=latest loads 17000 on first launch and 17500+ on any crash-resume
# (skip_dataloader_dcp=true -> resume is world-size agnostic; only random_states_0.pkl exists, tolerated).
# HELIOS_THROTTLE_FREE=1 skips the per-step free_memory() stall (validated -26% step time); if an OOM
# ever appears, drop that export and resubmit.
# Launch:  sbatch scripts/training/train_stage1_lora_cfr_368_correct.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_lora_cfr_368_correct.yaml}

# Seed the new output_dir with a symlink to checkpoint-17000 so resume_from_checkpoint=latest finds it.
# NOTE: link ONLY checkpoint-17000 (not checkpoint-17000-final: the 'latest' parser int()s the 2nd
# dash-field, so both would tie at 17000 and the pick would be nondeterministic).
OUTDIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368_correct
SRC17000=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368/checkpoint-17000
mkdir -p "$OUTDIR"
if [ ! -e "$OUTDIR/checkpoint-17000" ]; then
  ln -sfn "$SRC17000" "$OUTDIR/checkpoint-17000"
  echo "seeded $OUTDIR/checkpoint-17000 -> $SRC17000"
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29517
export WORLD_SIZE=$SLURM_NTASKS
echo "train_lora368c: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

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
  export HELIOS_FORCE_LR=1
  export HELIOS_THROTTLE_FREE=1
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_lora368c_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_lora368c_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_lora368c_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_LORA368C DONE $(date +%s)"

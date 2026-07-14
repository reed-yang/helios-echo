#!/bin/bash
#SBATCH --job-name=helios_rwtag
#SBATCH --nodes=1
#SBATCH --nodelist=c-node07
#SBATCH --ntasks-per-node=8            # one task per GPU; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_rwtag_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_rwtag_%j.out
# Continue-train the stage1 LoRA from stage1_lora_cfr_368_correct/checkpoint-19500 on the TAGGED
# reweight_v2 mix (symlink tree /mnt/beegfs/xiangbo/helios_data/reweight_v2/, built by
# scripts/data/build_reweight_tree_v2.py): realsource x1, music x2, animal x10, jesus x30,
# genesis x50, hatching(+extension) x200 = 154,015 samples, ONE true epoch.
# Every caption gets a leading dataset tag ("<hatching*>" etc.): HELIOS_TAG_EMBED_DIR points the
# dataloader at re-encoded UMT5 sidecars (scripts/data/encode_tag_sidecars.py) - the .pt latents
# are shared with the untouched originals, nothing is copied.
# SINGLE NODE c-node07 (its cross-node NCCL wedges; in-node NVLink is fine, node otherwise idle).
# GLOBAL BATCH 120 = 8 GPU x bs3 x accum5. 19500 + 1284 -> 20784 (checkpoint-20784-final).
# mc-node02 + c-node08 are left for the live checkpoint eval (eval_rwtag_watch.sbatch).
# Launch:  sbatch scripts/training/train_stage1_lora_rwtag.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_lora_rwtag_368.yaml}

# Seed the new output_dir with a symlink to checkpoint-19500 so resume_from_checkpoint=latest finds it.
OUTDIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_rwtag_368
SRC19500=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368_correct/checkpoint-19500
mkdir -p "$OUTDIR"
if [ ! -e "$OUTDIR/checkpoint-19500" ]; then
  ln -sfn "$SRC19500" "$OUTDIR/checkpoint-19500"
  echo "seeded $OUTDIR/checkpoint-19500 -> $SRC19500"
fi

# Refuse to start before the tagged sidecar store is complete (dataloader raises per-sample
# on a missing sidecar, which would spin the retry loop instead of failing the job cleanly).
NSC=$(ls /mnt/beegfs/xiangbo/helios_data/reweight_v2_tagembed 2>/dev/null | wc -l)
if [ "$NSC" -lt 80696 ]; then
  echo "FATAL: only $NSC/80696 tag sidecars present - run scripts/data/encode_tag_sidecars.sbatch first"
  exit 1
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29521
export WORLD_SIZE=$SLURM_NTASKS
echo "train_rwtag: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

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
  export HELIOS_TAG_EMBED_DIR=/mnt/beegfs/xiangbo/helios_data/reweight_v2_tagembed
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_rwtag_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_rwtag_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_rwtag_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_RWTAG DONE $(date +%s)"

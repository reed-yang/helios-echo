#!/bin/bash
#SBATCH --job-name=helios_rw368
#SBATCH --nodes=5
#SBATCH --nodelist=mc-node01,c-node03,c-node04,c-node05,c-node06
#SBATCH --ntasks-per-node=8            # one task per GPU -> 40 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_rw368_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_rw368_%j.out
# Continue-train the stage1 LoRA from stage1_lora_cfr_368_correct/checkpoint-19500 on the REWEIGHTED
# multi-source mix (symlink-materialized under /mnt/beegfs/xiangbo/helios_data/reweight_v1/, built by
# build_reweight_tree.py): hatching x200, genesis x60, jesus x20, animal x10, sekai x3, spatialVID x1,
# everything else x2. One true epoch = 1,396,442 samples.
# GLOBAL BATCH 120 = 5 nodes x 8 GPU x bs3 x accum1. 19500 + 11640 -> 31140. (Was 6 nodes/gb144;
# c-node07 dropped 2026-07-05: its cross-node NCCL wedges on the first collective — confirmed by a
# clean 2-node allreduce test vs c-node04 — even though IB links/modules look healthy.)
# mc-node02 + c-node08 are left free for the live checkpoint eval (eval_reweight_watch.sbatch).
# output_dir is SEEDED below with a symlink checkpoint-19500 -> .../stage1_lora_cfr_368_correct/
# checkpoint-19500, so resume_from_checkpoint=latest loads 19500 on first launch and 20000+ on any
# crash-resume (skip_dataloader_dcp=true -> resume is world-size agnostic).
# HELIOS_FORCE_LR=1 re-applies config lr 1e-5 over the restored optimizer state (idempotent here
# since ckpt-19500 was already saved at 1e-5, but keeps the config authoritative).
# Launch:  sbatch scripts/training/train_stage1_lora_reweight.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_lora_reweight_368.yaml}

# Seed the new output_dir with a symlink to checkpoint-19500 so resume_from_checkpoint=latest finds it.
OUTDIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_reweight_368
SRC19500=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368_correct/checkpoint-19500
mkdir -p "$OUTDIR"
if [ ! -e "$OUTDIR/checkpoint-19500" ]; then
  ln -sfn "$SRC19500" "$OUTDIR/checkpoint-19500"
  echo "seeded $OUTDIR/checkpoint-19500 -> $SRC19500"
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29519
export WORLD_SIZE=$SLURM_NTASKS
echo "train_rw368: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

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
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_rw368_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_rw368_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_rw368_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_RW368 DONE $(date +%s)"

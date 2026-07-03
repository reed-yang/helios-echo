#!/bin/bash
#SBATCH --job-name=helios_l368c_sm
#SBATCH --nodes=1
#SBATCH --nodelist=mc-node02           # idle; keeps the 5 target nodes + busy c-node07 untouched
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_l368c_sm_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_l368c_sm_%j.out
# SMOKE for the correct-recipe continue run (docs/HELIOS_CORRECT_RECIPE.md §5.3). Validates on 1 node:
#   resume from stage1_lora_cfr_368/checkpoint-17000, HELIOS_FORCE_LR=1 drops lr 3e-5 -> 1e-5,
#   forced i2v-drop / downsample(blur) / saturation branches all fire ([DBG-BRANCH] lines), finite loss.
# Launch:  sbatch scripts/training/train_stage1_lora_cfr_368_correct_smoke.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_lora_cfr_368_correct_smoke.yaml}

rm -rf /mnt/beegfs/xiangbo/helios_runs/_smoke368c_out
mkdir -p /mnt/beegfs/xiangbo/helios_runs/_smoke368c_out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29513
export WORLD_SIZE=$SLURM_NTASKS
echo "train_l368c_sm: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

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
  export HELIOS_FORCE_LR=1
  export HELIOS_DEBUG_BRANCH=1
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_l368c_sm_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_l368c_sm_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_l368c_sm_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "SMOKE_L368C DONE $(date +%s)"

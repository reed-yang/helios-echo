#!/bin/bash
#SBATCH --job-name=helios_sftv2
#SBATCH --nodes=4
#SBATCH --nodelist=mc-node01,mc-node02,c-node03,c-node04   # user-reserved 4 nodes = 32 GPU
#SBATCH --ntasks-per-node=1          # ONE accelerate launch per node; it spawns 8 local procs itself
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_sftv2_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_sftv2_%j.out
# FULL fine-tune (no LoRA) of the 14B Helios transformer on the cfr_int / captions_30b_32f corpus.
# DeepSpeed ZeRO-2 via `accelerate launch --config_file multi_node_example_zero2.yaml` (the plain
# env-launcher = DDP and won't fit 14B full params + Adam). One srun task per node spawns 8 local
# ranks; machine_rank=$SLURM_NODEID. GLOBAL BATCH 32 = 4 x 8 x bs1 x accum1. Idempotent restart:
# resume_from_checkpoint=latest (DeepSpeed ckpts are world-size locked to 32 -> resume on 32 GPU).
#
#   sbatch scripts/training/train_stage1_fullft_cfr.sbatch                         # full run
#   sbatch --nodes=1 --nodelist=c-node05 CONFIG=scripts/training/configs/stage1_fullft_cfr_smoke.yaml \
#          scripts/training/train_stage1_fullft_cfr.sbatch                         # 8-GPU smoke
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_sft_cfr_v2.yaml}
# DeepSpeed ZeRO-2 WITH a deepspeed-managed LR scheduler (WarmupLR). The scheduler must live in the
# deepspeed config so its state is saved/restored inside the engine checkpoint; otherwise resume
# crashes looking for a scheduler.bin that save_state never writes under deepspeed.
ACC_CFG=${ACC_CFG:-scripts/accelerate_configs/multi_node_zero2_sched_2e6.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29506
NNODES=$SLURM_NNODES
NPROC=$(( NNODES * 8 ))
echo "train_sftv2: nodes=$NNODES nproc=$NPROC master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG acc=$ACC_CFG nodelist=$SLURM_JOB_NODELIST"

srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=online
  export WANDB_API_KEY=$(cat /mnt/beegfs/xiangbo/.wandb_key 2>/dev/null)
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  # node-local Triton/inductor cache + gcc serialization (defeats the beegfs concurrent-header race).
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_ft_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_ft_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_ft_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  accelerate launch \
    --config_file '"$ACC_CFG"' \
    --num_machines '"$NNODES"' \
    --num_processes '"$NPROC"' \
    --machine_rank $SLURM_NODEID \
    --main_process_ip '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_FULLFT DONE $(date +%s)"

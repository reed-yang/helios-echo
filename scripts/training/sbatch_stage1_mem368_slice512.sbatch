#!/bin/bash
#SBATCH -p gpu
#SBATCH --exclude=c-node03,c-node04,c-node06,c-node07,c-node08
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --job-name=stageA_slice512
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/stageA_slice512_run3.log
# Queued without -w: Slurm serializes us behind visible jobs on the allowed
# nodes. Exclusions: c-node04/06 pinghe (must stay undisturbed), c-node07
# rwtag, c-node08 foreign non-Slurm load + unresponsive, c-node03 drained.
# Preflight guards against colleagues' non-Slurm GPU processes (invisible to
# Slurm): bail out loudly instead of stacking into an OOM.
foreign=$(/usr/bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [ "$foreign" -gt 0 ]; then
  echo "PREFLIGHT_ABORT: $foreign foreign GPU processes on $(hostname); refusing to stack"
  exit 42
fi
cd /mnt/beegfs/siyuan/workspace/helios-echo
export WANDB_MODE=offline
export PYTHONPATH=.
/mnt/beegfs/yuheng/miniconda3/envs/helios/bin/accelerate launch --num_processes 8 \
  --mixed_precision bf16 train_helios.py \
  --config scripts/training/configs/stage1_lora_mem368_A_slice512.yaml
echo "EXIT_CODE=$?"

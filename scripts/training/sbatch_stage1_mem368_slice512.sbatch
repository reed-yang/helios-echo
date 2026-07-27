#!/bin/bash
#SBATCH -p gpu
#SBATCH -w c-node08
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --job-name=stageA_slice512
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/stageA_slice512_run2.log
# Stage A slice512 run — sbatch so the job is decoupled from any login session.
# Node etiquette: verify the target node has zero foreign GPU processes BEFORE
# submitting (Slurm cannot see colleagues' non-Slurm jobs):
#   ssh <node> nvidia-smi --query-compute-apps=pid,used_memory --format=csv
cd /mnt/beegfs/siyuan/workspace/helios-echo
export WANDB_MODE=offline
export PYTHONPATH=.
/mnt/beegfs/yuheng/miniconda3/envs/helios/bin/accelerate launch --num_processes 8 \
  --mixed_precision bf16 train_helios.py \
  --config scripts/training/configs/stage1_lora_mem368_A_slice512.yaml
echo "EXIT_CODE=$?"

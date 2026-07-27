#!/bin/bash
#SBATCH -p gpu
##SBATCH -w <SET-NODE-AFTER-OWNERSHIP-CHECK>
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --job-name=stageA_pilot_resume
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/stageA_pilot_run2.log
# Resumes the pilot from checkpoint-500 (resume_from_checkpoint: "latest").
# DO NOT submit blindly: Slurm cannot see non-Slurm colleague jobs — run the
# nvidia-smi ownership check on the target node first, then set -w above.
cd /mnt/beegfs/siyuan/workspace/helios-echo
export WANDB_MODE=offline
export PYTHONPATH=.
/mnt/beegfs/yuheng/miniconda3/envs/helios/bin/accelerate launch --num_processes 8 \
  --mixed_precision bf16 train_helios.py \
  --config scripts/training/configs/stage1_lora_mem368_A_pilot.yaml
echo "EXIT_CODE=$?"

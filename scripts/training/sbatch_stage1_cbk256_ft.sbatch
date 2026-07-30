#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --job-name=stage1_cbk256_ft
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/stage1_cbk256_ft_run1.log

# Single-node K-means-codebook history finetune: bs2 x accum2 x 8 = global batch 32.
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
  --config scripts/training/configs/stage1_lora_cbk256_ft.yaml
rc=$?
echo "EXIT_CODE=$rc"
exit $rc

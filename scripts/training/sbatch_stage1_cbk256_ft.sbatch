#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:7
#SBATCH --nodelist=mc-node02
#SBATCH --cpus-per-task=84
#SBATCH --time=24:00:00
#SBATCH --job-name=stage1_cbk256_ft
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/stage1_cbk256_ft_run1.log

# Single-node K-means-codebook history finetune on the 7 temperature-normal
# cards of mc-node02: bs2 x accum2 x 7 = global batch 28. Stage A ran at 32
# (2 x 1 x 16); 32 is not reachable with 7 processes at an integer accumulation,
# so this run is at 28 and the difference is recorded rather than hidden.
# c-node08 cannot host this: jiongze's non-Slurm processes leave 5-64GB free per
# card there while this recipe needs roughly 120GB.
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
/mnt/beegfs/yuheng/miniconda3/envs/helios/bin/accelerate launch --num_processes 7 \
  --mixed_precision bf16 train_helios.py \
  --config scripts/training/configs/stage1_lora_cbk256_ft.yaml
rc=$?
echo "EXIT_CODE=$rc"
exit $rc

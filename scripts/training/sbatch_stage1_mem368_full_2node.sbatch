#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodelist=c-node05,mc-node01
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --job-name=stageA_full
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/stageA_full_run1.log

# Full-corpus Stage A: 168,431 clips, 12k steps, 2 nodes x 8 H200 = 16 GPUs.
# Effective hyperparams match the completed pilot/slice512 recipe: global batch
# bs2 x accum1 x 16 = 32 (accum halved to keep step semantics comparable while
# doubling throughput). resume_from_checkpoint=latest -> resubmit continues
# after the 48h limit. One accelerate launcher per node; NCCL over IB.

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29671
echo "NODES=$SLURM_JOB_NODELIST master=$MASTER_ADDR"

srun --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  foreign=$(/usr/bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$foreign" -gt 0 ]; then
    echo "PREFLIGHT_ABORT: $foreign foreign GPU processes on $(hostname); refusing to stack"
    exit 42
  fi
  cd /mnt/beegfs/siyuan/workspace/helios-echo
  export WANDB_MODE=offline
  export PYTHONPATH=.
  export NCCL_DEBUG=WARN
  echo "RANK_NODE=$SLURM_NODEID host=$(hostname)"
  /mnt/beegfs/yuheng/miniconda3/envs/helios/bin/accelerate launch \
    --num_machines 2 --num_processes 16 \
    --machine_rank "$SLURM_NODEID" \
    --main_process_ip "'"$MASTER_ADDR"'" --main_process_port "'"$MASTER_PORT"'" \
    --mixed_precision bf16 train_helios.py \
    --config scripts/training/configs/stage1_lora_mem368_A_full.yaml
'
rc=$?
echo "EXIT_CODE=$rc"
exit $rc

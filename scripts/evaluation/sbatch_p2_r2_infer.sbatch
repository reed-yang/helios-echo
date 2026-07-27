#!/bin/bash
#SBATCH --job-name=p2r2_infer
#SBATCH --partition=gpu
#SBATCH --nodelist=c-node08
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/p2_interim/logs/%x_%j.log

# P2-interim r2: memory-on arm with Stage A trained memory weights.
# Usage: sbatch sbatch_p2_r2_infer.sbatch <memory_partial.pth> <prompt_index> <run_id>
# Deliberately stacked on partially occupied GPUs (user-authorized); no foreign-proc guard.
set -uo pipefail
PARTIAL="$1"
PIDX="$2"
RUNID="$3"

cd /mnt/beegfs/siyuan/workspace/helios-echo
ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios

echo "JOB=$SLURM_JOB_ID node=$(hostname) cvd=${CUDA_VISIBLE_DEVICES:-unset} partial=$PARTIAL pidx=$PIDX runid=$RUNID"
/usr/bin/nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

PYTHONPATH=. "$ENV/bin/python" scripts/evaluation/run_p2_interim_drift_ab.py \
  --arm on \
  --prompt-index "$PIDX" \
  --run-id "$RUNID" \
  --memory-partial "$PARTIAL"
rc=$?
echo "EXIT_CODE=$rc"
exit $rc

#!/bin/bash
#SBATCH --job-name=p2r3_infer
#SBATCH --partition=gpu
#SBATCH --nodelist=c-node08
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/p2_interim/logs/%x_%j.log

# P2 r3: rep50 structural prompts, three-arm rebuild (off / on-untrained / on-trained).
# Usage: sbatch sbatch_p2_r3_infer.sbatch <arm> <prompt_index> <run_id> [memory_partial.pth|none]
# Deliberately stacked on partially occupied GPUs (user-authorized).
set -uo pipefail
ARM="$1"
PIDX="$2"
RUNID="$3"
PARTIAL="${4:-none}"

cd /mnt/beegfs/siyuan/workspace/helios-echo
ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios

echo "JOB=$SLURM_JOB_ID node=$(hostname) cvd=${CUDA_VISIBLE_DEVICES:-unset} arm=$ARM pidx=$PIDX runid=$RUNID partial=$PARTIAL"
/usr/bin/nvidia-smi --query-gpu=index,compute_mode,memory.used --format=csv,noheader || true

# Compute nodes leak an unmanaged Prohibited-mode GPU into every job cgroup
# (findings 2026-07-27), and colleagues' stacked processes grow dynamically
# (2026-07-28: 63GB -> 116GB mid-run OOMed job 5739). Pick the Default-mode
# visible GPU with the most free memory and require ~60GB headroom.
goodidx=$(/usr/bin/nvidia-smi --query-gpu=index,compute_mode,memory.free --format=csv,noheader,nounits \
  | /usr/bin/awk -F', ' '$2 == "Default" && $3 > best {best = $3; idx = $1} END {if (best >= 60000) print idx}')
if [ -z "$goodidx" ]; then
  echo "NO_GPU_WITH_HEADROOM"
  echo "EXIT_CODE=43"
  exit 43
fi
export CUDA_VISIBLE_DEVICES="$goodidx"
echo "REMAPPED cvd=$CUDA_VISIBLE_DEVICES"

ARGS=(--arm "$ARM" --prompt-index "$PIDX" --run-id "$RUNID" --prompt-set rep50 --num-prompts 8)
if [ "$PARTIAL" != "none" ]; then
  ARGS+=(--memory-partial "$PARTIAL")
fi

PYTHONPATH=. "$ENV/bin/python" scripts/evaluation/run_p2_interim_drift_ab.py "${ARGS[@]}"
rc=$?
echo "EXIT_CODE=$rc"
exit $rc

#!/bin/bash
#SBATCH --job-name=p2r4_evsw
#SBATCH --partition=gpu
#SBATCH --nodelist=c-node08
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/p2_interim/logs/%x_%j.log

# P2 r4: rep50 event-switch protocol — 6 segments x 7 sections, hard switch.
# Usage: sbatch sbatch_p2_r4_eventswitch.sbatch <arm> <prompt_index> <run_id> [memory_partial.pth|none]
set -uo pipefail
ARM="$1"
PIDX="$2"
RUNID="$3"
PARTIAL="${4:-none}"

cd /mnt/beegfs/siyuan/workspace/helios-echo
ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios

echo "JOB=$SLURM_JOB_ID node=$(hostname) cvd=${CUDA_VISIBLE_DEVICES:-unset} arm=$ARM pidx=$PIDX runid=$RUNID partial=$PARTIAL"
/usr/bin/nvidia-smi --query-gpu=index,compute_mode,memory.free --format=csv,noheader || true

# pinghe's runs live outside Slurm, so sinfo reports their whole nodes as idle
# and their ~86GB of free VRAM per H200 passes the headroom check below. The
# --nodelist pin is not a guarantee (a -w override reaches a different node),
# so refuse the node explicitly rather than stacking on them.
owners=$(/usr/bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader \
  | /usr/bin/xargs -r -n1 /bin/ps -o user= -p 2>/dev/null | /usr/bin/sort -u | /usr/bin/tr '\n' ',')
case ",$owners" in
  *,pinghe,*)
    echo "FOREIGN_OWNER_REFUSED host=$(hostname) owners=$owners"
    echo "EXIT_CODE=44"
    exit 44
    ;;
esac
echo "GPU_OWNERS=$owners"

# Pick the Default-mode visible GPU with the most free memory (leaked
# Prohibited cards + colleagues' dynamically growing processes; see findings).
goodidx=$(/usr/bin/nvidia-smi --query-gpu=index,compute_mode,memory.free --format=csv,noheader,nounits \
  | /usr/bin/awk -F', ' '$2 == "Default" && $3 > best {best = $3; idx = $1} END {if (best >= 60000) print idx}')
if [ -z "$goodidx" ]; then
  echo "NO_GPU_WITH_HEADROOM"
  echo "EXIT_CODE=43"
  exit 43
fi
export CUDA_VISIBLE_DEVICES="$goodidx"
echo "REMAPPED cvd=$CUDA_VISIBLE_DEVICES"

ARGS=(--arm "$ARM" --prompt-index "$PIDX" --run-id "$RUNID" \
      --prompt-set rep50 --num-prompts 8 --segments 6 --sections-per-segment 7)
if [ "$PARTIAL" != "none" ]; then
  ARGS+=(--memory-partial "$PARTIAL")
fi

PYTHONPATH=. "$ENV/bin/python" scripts/evaluation/run_p2_interim_drift_ab.py "${ARGS[@]}"
rc=$?
echo "EXIT_CODE=$rc"
exit $rc

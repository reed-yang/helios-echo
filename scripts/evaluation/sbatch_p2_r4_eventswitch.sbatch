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
# Usage: sbatch sbatch_p2_r4_eventswitch.sbatch <arm> <prompt_index> <run_id> \
#          [memory_partial.pth|none] [num_prompts]
# num_prompts widens the rep50 case window: the default 8 keeps every existing
# campaign identical; 16 exposes cases 8-15 for held-out generalization.
set -uo pipefail
ARM="$1"
PIDX="$2"
RUNID="$3"
PARTIAL="${4:-none}"
NUMP="${5:-8}"
PROJ="${6:-none}"
PROJ_ALPHA="${7:-1.0}"
PROJ_STEP="${8:-0}"
PROJ_CODEBOOK="${9:-none}"

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

# Slurm's own CUDA_VISIBLE_DEVICES is unusable on these nodes: an unmanaged
# Prohibited-mode card leaks into every job cgroup and takes index 0 (findings
# 2026-07-27, verified again 2026-07-30 by a torch probe returning
# AcceleratorError). So each job must choose a card itself — and if every job
# just takes "the one with the most free memory", simultaneous jobs all choose
# the SAME card: observed 7 jobs on 4 cards with 2 idle, and one card driven to
# 132/143GB, which is how job 5739 OOMed. Claim a card with an exclusive lock so
# our own jobs on this node never land on the same one. The lock is held for the
# life of the job through the open descriptor.
# The claim runs in THIS shell, not a subshell: a lock taken inside $(...) dies
# with the subshell and would guarantee nothing.
candidates=$(/usr/bin/nvidia-smi --query-gpu=index,compute_mode,memory.free --format=csv,noheader,nounits \
  | /usr/bin/awk -F', ' '$2 == "Default" && $3 >= 60000 {print $3, $1}' | /usr/bin/sort -rn | /usr/bin/awk '{print $2}')
if [ -z "$candidates" ]; then
  echo "NO_GPU_WITH_HEADROOM"
  echo "EXIT_CODE=43"
  exit 43
fi
goodidx=""
for idx in $candidates; do
  exec {GPU_LOCK_FD}>"/tmp/helios_gpu_${idx}.lock" 2>/dev/null || continue
  if /usr/bin/flock -n "$GPU_LOCK_FD"; then
    goodidx="$idx"
    break
  fi
  exec {GPU_LOCK_FD}>&-
done
if [ -n "$goodidx" ]; then
  export CUDA_VISIBLE_DEVICES="$goodidx"
  echo "GPU_CLAIMED cvd=$CUDA_VISIBLE_DEVICES (exclusive lock, candidates=$(echo $candidates | tr '\n' ' '))"
else
  # Every roomy card is already claimed by one of our jobs: share the roomiest
  # rather than failing, and say so in the log.
  export CUDA_VISIBLE_DEVICES=$(echo "$candidates" | /usr/bin/head -1)
  echo "GPU_SHARED cvd=$CUDA_VISIBLE_DEVICES (all free cards already locked)"
fi

ARGS=(--arm "$ARM" --prompt-index "$PIDX" --run-id "$RUNID" \
      --prompt-set rep50 --num-prompts "$NUMP" --segments 6 --sections-per-segment 7)
if [ "$PARTIAL" != "none" ]; then
  ARGS+=(--memory-partial "$PARTIAL")
fi
if [ "$PROJ" != "none" ]; then
  ARGS+=(--history-projection "$PROJ" --history-projection-alpha "$PROJ_ALPHA")
  if [ "$PROJ" = "quantize" ]; then
    ARGS+=(--history-projection-step "$PROJ_STEP")
  fi
  if [ "$PROJ" = "codebook" ]; then
    ARGS+=(--history-projection-codebook "$PROJ_CODEBOOK")
  fi
fi

PYTHONPATH=. "$ENV/bin/python" scripts/evaluation/run_p2_interim_drift_ab.py "${ARGS[@]}"
rc=$?
echo "EXIT_CODE=$rc"
exit $rc

#!/bin/bash
# Run one rollout OUTSIDE Slurm, on a node whose GPUs Slurm has already handed
# to someone else. Slurm gives c-node05 and mc-node01 entirely to yuheng's
# whole-node jobs, but each card there keeps ~88GB free, which is enough for one
# rollout (~60GB). Stacking on yuheng is explicitly allowed; stacking on pinghe
# is not, so the owner check stays.
#
# Unlike a Slurm job, this process sees the node's REAL GPU indices with no
# cgroup in the way, so an exclusive lock per card genuinely keeps our own
# rollouts off each other's GPU.
#
# Usage (run on the target node, or via ssh):
#   bash_infer_worker.sh <arm> <prompt_index> <run_id> [partial|none] [sections] \
#       [projection] [alpha] [step] [codebook|none] [num_prompts] [site] [smooth]
set -uo pipefail

ARM="$1"
PIDX="$2"
RUNID="$3"
PARTIAL="${4:-none}"
SECTIONS="${5:-349}"
PROJ="${6:-none}"
PROJ_ALPHA="${7:-1.0}"
PROJ_STEP="${8:-0}"
PROJ_CODEBOOK="${9:-none}"
NUMP="${10:-8}"
PROJ_SITE="${11:-read}"
PROJ_SMOOTH="${12:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-65000}"

REPO=/mnt/beegfs/siyuan/workspace/helios-echo
ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
cd "$REPO"
LOGDIR="$REPO/results/p2_interim/logs"
mkdir -p "$LOGDIR"

owners=$(/usr/bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader \
  | /usr/bin/xargs -r -n1 /bin/ps -o user= -p 2>/dev/null | /usr/bin/sort -u | /usr/bin/tr '\n' ',')
case ",$owners" in
  *,pinghe,*)
    echo "FOREIGN_OWNER_REFUSED host=$(hostname) owners=$owners"
    echo "EXIT_CODE=44"
    exit 44
    ;;
esac

candidates=$(/usr/bin/nvidia-smi --query-gpu=index,compute_mode,memory.free --format=csv,noheader,nounits \
  | /usr/bin/awk -F', ' -v m="$MIN_FREE_MIB" '$2 == "Default" && $3 >= m {print $3, $1}' \
  | /usr/bin/sort -rn | /usr/bin/awk '{print $2}')
if [ -z "$candidates" ]; then
  echo "NO_GPU_WITH_HEADROOM host=$(hostname) need=${MIN_FREE_MIB}MiB owners=$owners"
  echo "EXIT_CODE=43"
  exit 43
fi

# The lock is held for the life of this shell through the open descriptor, so it
# releases when the rollout exits. Taking it inside $(...) would release it
# immediately and guarantee nothing.
goodidx=""
for idx in $candidates; do
  exec {GPU_LOCK_FD}>"/tmp/helios_bash_gpu_${idx}.lock" 2>/dev/null || continue
  if /usr/bin/flock -n "$GPU_LOCK_FD"; then
    goodidx="$idx"
    break
  fi
  exec {GPU_LOCK_FD}>&-
done
if [ -z "$goodidx" ]; then
  echo "ALL_CANDIDATE_GPUS_CLAIMED host=$(hostname) candidates=$(echo $candidates | tr '\n' ' ')"
  echo "EXIT_CODE=45"
  exit 45
fi
export CUDA_VISIBLE_DEVICES="$goodidx"

ARGS=(--arm "$ARM" --prompt-index "$PIDX" --run-id "$RUNID" --prompt-set rep50 --num-prompts "$NUMP" --sections "$SECTIONS")
if [ "$PARTIAL" != "none" ]; then
  ARGS+=(--memory-partial "$PARTIAL")
fi
if [ "$PROJ" != "none" ]; then
  ARGS+=(--history-projection "$PROJ" --history-projection-alpha "$PROJ_ALPHA" --history-projection-site "$PROJ_SITE")
  if [ "$PROJ_SMOOTH" != "0" ]; then
    ARGS+=(--history-projection-smooth "$PROJ_SMOOTH")
  fi
  if [ "$PROJ" = "quantize" ]; then
    ARGS+=(--history-projection-step "$PROJ_STEP")
  fi
  if [ "$PROJ" = "codebook" ]; then
    ARGS+=(--history-projection-codebook "$PROJ_CODEBOOK")
  fi
fi

LOG="$LOGDIR/bash_${RUNID}_${ARM}_p${PIDX}_$(hostname -s)_gpu${goodidx}.log"
{
  echo "BASH_JOB host=$(hostname) gpu=$goodidx free_required=${MIN_FREE_MIB}MiB owners=$owners"
  echo "ARGS: ${ARGS[*]}"
} > "$LOG"
# The log keeps the same EXIT_CODE= convention as the sbatch launchers so the
# existing watchers and metrics batches need no special case.
PYTHONPATH=. "$ENV/bin/python" scripts/evaluation/run_p2_interim_drift_ab.py "${ARGS[@]}" >> "$LOG" 2>&1
rc=$?
echo "EXIT_CODE=$rc" >> "$LOG"
echo "BASH_DONE host=$(hostname) gpu=$goodidx rc=$rc log=$LOG"
exit $rc

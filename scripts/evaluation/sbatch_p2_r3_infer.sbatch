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
# Usage: sbatch sbatch_p2_r3_infer.sbatch <arm> <prompt_index> <run_id> \
#          [memory_partial.pth|none] [sections] [projection] [alpha] [step] [codebook.pt|none]
# projection: none | renorm | codebook | quantize (training-free anti-drift, 2026-07-29 design)
# Deliberately stacked on partially occupied GPUs (user-authorized).
set -uo pipefail
ARM="$1"
PIDX="$2"
RUNID="$3"
PARTIAL="${4:-none}"
SECTIONS="${5:-66}"
PROJ="${6:-none}"
PROJ_ALPHA="${7:-1.0}"
PROJ_STEP="${8:-0}"
PROJ_CODEBOOK="${9:-none}"
# Widens the rep50 case window; the default 8 keeps every existing campaign
# identical, 16 exposes the held-out cases 8-15.
NUMP="${10:-8}"

cd /mnt/beegfs/siyuan/workspace/helios-echo
ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios

echo "JOB=$SLURM_JOB_ID node=$(hostname) cvd=${CUDA_VISIBLE_DEVICES:-unset} arm=$ARM pidx=$PIDX runid=$RUNID partial=$PARTIAL"
/usr/bin/nvidia-smi --query-gpu=index,compute_mode,memory.used --format=csv,noheader || true

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

# Slurm hands a single-GPU job its own card as cgroup index 0, so its own
# CUDA_VISIBLE_DEVICES is both correct and distinct per job. Probe it first and
# only fall back to picking a card ourselves. An UNCONDITIONAL "pick the
# Default-mode card with the most free memory" makes every job that starts at
# the same moment select the SAME card: observed 2026-07-30 on mc-node02 with 7
# jobs piled onto 4 cards while 2 sat idle. The fallback still exists because
# these nodes leak an unmanaged Prohibited-mode card into the cgroup (findings
# 2026-07-27), which makes the assigned device unusable, and because a
# colleague's stacked process can grow mid-run (2026-07-28 OOM of job 5739).
probe=$("$ENV/bin/python" -c "
import torch
try:
    free, _ = torch.cuda.mem_get_info(0)
    print('OK', free // (1024 * 1024))
except Exception as exc:
    print('FAIL', type(exc).__name__)
" 2>/dev/null)
echo "GPU_PROBE=$probe"
keep=0
case "$probe" in
  OK*) free_mib=${probe#OK }; [ "$free_mib" -ge 60000 ] && keep=1 ;;
esac
if [ "$keep" = "1" ]; then
  echo "GPU_KEEP slurm_cvd=${CUDA_VISIBLE_DEVICES:-unset} free=${free_mib}MiB"
else
  goodidx=$(/usr/bin/nvidia-smi --query-gpu=index,compute_mode,memory.free --format=csv,noheader,nounits \
    | /usr/bin/awk -F', ' '$2 == "Default" && $3 > best {best = $3; idx = $1} END {if (best >= 60000) print idx}')
  if [ -z "$goodidx" ]; then
    echo "NO_GPU_WITH_HEADROOM probe=$probe"
    echo "EXIT_CODE=43"
    exit 43
  fi
  export CUDA_VISIBLE_DEVICES="$goodidx"
  echo "REMAPPED cvd=$CUDA_VISIBLE_DEVICES (fallback, probe=$probe)"
fi

ARGS=(--arm "$ARM" --prompt-index "$PIDX" --run-id "$RUNID" --prompt-set rep50 --num-prompts "$NUMP" --sections "$SECTIONS")
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

#!/bin/bash
# Dedicated GPU-waste checker: for each of the user's RUNNING helios Slurm jobs, sample per-GPU
# util/mem (inside the job's cgroup via `srun --overlap`) and flag idle / under-filled GPUs so we
# never leave GPUs wasting on node7/node8. Print a one-line-per-GPU report with LOW-UTIL / LOW-MEM flags.
#   usage: bash scripts/wan22/gpu_watch.sh            # single sample
#          bash scripts/wan22/gpu_watch.sh 3 20       # 3 samples, 20s apart (avg over transient stalls)
SAMPLES=${1:-2}; GAP=${2:-15}
UTIL_LO=${UTIL_LO:-25}      # flag GPUs below this % util
MEM_LO_FRAC=${MEM_LO_FRAC:-30}  # flag GPUs using < this % of memory

jobs=$(squeue -u "$USER" -h -t RUNNING -o "%i %j" | grep -iE "wan22|helios" | awk '{print $1}')
[ -z "$jobs" ] && { echo "[gpu_watch] no running helios jobs"; exit 0; }

for jid in $jobs; do
  info=$(squeue -j "$jid" -h -o "%j on %N (%M)")
  echo "== job $jid  $info =="
  # average util over SAMPLES snapshots (util dips between steps; averaging shows the real picture)
  for s in $(seq 1 "$SAMPLES"); do
    srun --overlap --jobid="$jid" nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
    [ "$s" -lt "$SAMPLES" ] && sleep "$GAP"
  done | awk -F', ' -v ul="$UTIL_LO" -v mf="$MEM_LO_FRAC" '
    { i=$1+0; u[$1]+=$2+0; m[$1]=$3+0; t[$1]=$4+0; n[$1]++ }
    END {
      tot=0; low=0
      for (g in u) {
        au=u[g]/n[g]; memf=(t[g]>0? m[g]*100/t[g] : 0)
        f=""; if (au<ul) f=f" LOW-UTIL"; if (memf<mf) f=f" LOW-MEM"
        printf "  gpu%-2s avg-util=%5.1f%%  mem=%6dMiB (%4.1f%%)%s\n", g, au, m[g], memf, f
        tot++; if (f!="") low++
      }
      printf "  -> %d/%d GPUs flagged (util<%s%% or mem<%s%%)\n", low, tot, ul, mf
    }'
done

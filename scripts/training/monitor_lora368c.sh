#!/bin/bash
# Crash-resume monitor for the lora368-correct run (recipe continue-train from ckpt-17000),
# HARDENED against resubmit storms (same pattern as monitor_fullbase368.sh):
#  - resubmits by NAME on crash (resume_from_checkpoint=latest; outdir seeded w/ checkpoint-17000)
#  - MIN_INTERVAL: never resubmit more than once per 25 min
#  - FAIL_CAP: consecutive no-progress resubmits -> stop and leave a loud note
# Launch detached:  setsid nohup bash scripts/training/monitor_lora368c.sh >/dev/null 2>&1 &
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
JOBNAME=helios_lora368c
SBATCH=scripts/training/train_stage1_lora_cfr_368_correct.sbatch
OUTDIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368_correct
MAXSTEPS=22000
MIN_INTERVAL=1500
FAIL_CAP=3
LOG=/mnt/beegfs/xiangbo/helios_runs/logs/monitor_lora368c.log
cd "$REPO"
last_submit=0
fails=0
last_ckpt_seen=-1
ckpt_step() { ls -1 "$OUTDIR" 2>/dev/null | grep -oE 'checkpoint-[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1; }
now() { date +%s; }
echo "[$(date)] monitor start (hardened: min_interval=${MIN_INTERVAL}s fail_cap=$FAIL_CAP)" >> "$LOG"
while true; do
  sleep 300
  last=$(ckpt_step); last=${last:-0}
  if [ "$last" -ge "$MAXSTEPS" ]; then
    echo "[$(date)] reached $last >= $MAXSTEPS, done. stopping monitor." >> "$LOG"; break
  fi
  running=$(squeue -u "$USER" -h -o "%j %t" 2>/dev/null | awk -v j="$JOBNAME" '$1==j && ($2=="R"||$2=="PD"){print}' | wc -l)
  if [ "$running" -ge 1 ]; then
    echo "[$(date)] ok: $JOBNAME running/pending (last ckpt=$last)" >> "$LOG"
    continue
  fi
  if [ "$last" -gt "$last_ckpt_seen" ]; then fails=0; else fails=$((fails+1)); fi
  if [ "$fails" -ge "$FAIL_CAP" ]; then
    echo "[$(date)] GIVING UP: $FAIL_CAP consecutive resubmits with no progress (last ckpt=$last). Needs a human." >> "$LOG"
    break
  fi
  since=$(( $(now) - last_submit ))
  if [ "$since" -lt "$MIN_INTERVAL" ]; then
    echo "[$(date)] $JOBNAME down (last ckpt=$last) but only ${since}s since last submit (<${MIN_INTERVAL}); waiting." >> "$LOG"
    continue
  fi
  jid=$(sbatch --parsable "$SBATCH" 2>>"$LOG")
  last_submit=$(now); last_ckpt_seen=$last
  echo "[$(date)] resubmitted as $jid (fails=$fails, last ckpt=$last)" >> "$LOG"
done

#!/bin/bash
# Crash-resume monitor for the full-base368 run, HARDENED against resubmit storms.
#  - resubmits by NAME on crash (resume_from_checkpoint=latest)
#  - MIN_INTERVAL: never resubmit more than once per 25 min (startup is ~15 min; faster = a storm)
#  - FAIL_CAP: if this many consecutive resubmits make NO forward progress (checkpoint step doesn't
#    advance), STOP and leave a loud note instead of storming.
# The run is launched WITH HELIOS_THROTTLE_FREE=1 in the sbatch, so every resubmit keeps the speedup.
# Launch detached:  setsid nohup bash scripts/training/monitor_fullbase368.sh >/dev/null 2>&1 &
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
JOBNAME=helios_fullbase368
SBATCH=scripts/training/train_stage1_fullbase_cfr368.sbatch
OUTDIR=/mnt/beegfs/xiangbo/helios_runs/stage1_fullbase_cfr368
MAXSTEPS=6000
MIN_INTERVAL=1500        # seconds between resubmits (25 min)
FAIL_CAP=3               # consecutive no-progress resubmits before giving up
LOG=/mnt/beegfs/xiangbo/helios_runs/logs/monitor_fullbase368.log
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
  # not running -> consider resubmit
  # progress check: did the checkpoint advance since the last resubmit?
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

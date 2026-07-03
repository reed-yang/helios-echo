#!/bin/bash
# Skip-to-latest checkpoint watcher for the live eval node.
# Polls the training output_dir for checkpoint-<N> dirs and, whenever a NEWER one
# than the last evaluated appears, runs one eval round on it. Intermediate
# checkpoints that arrived while a round was running are SKIPPED (always evaluate
# only the newest), so eval rounds never queue up regardless of relative speed.
#
# Usage: watch_and_eval.sh   (configure via env below)
set -uo pipefail

REPO=${REPO:-/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team}
TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-/mnt/beegfs/xiangbo/helios_runs/stage1_continue_ohv}
STATE_FILE=${STATE_FILE:-/mnt/beegfs/xiangbo/helios_runs/eval_out/.last_evaluated_step}
POLL_INTERVAL=${POLL_INTERVAL:-120}   # seconds between checks
WHICH=${WHICH:-both}
# Per-round eval script: single-prompt run_eval_once.sh by default; the multi-event eval
# node overrides with run_eval_once_multievent.sh (same <ckpt> <step> <which> interface).
EVAL_RUN_SCRIPT=${EVAL_RUN_SCRIPT:-run_eval_once.sh}

mkdir -p "$(dirname "$STATE_FILE")"
last=$(cat "$STATE_FILE" 2>/dev/null || echo -1)
echo "[watch] watching $TRAIN_OUTPUT_DIR (last evaluated step=$last, poll=${POLL_INTERVAL}s)"

# Baseline (pre-fine-tune) eval of raw Helios-Base as step 0, once.
if [ "${RUN_BASELINE:-1}" = "1" ] && [ "$last" -lt 0 ]; then
  echo "[watch] running BASELINE eval (step 0, no LoRA)"
  if bash "$REPO/scripts/eval_live/$EVAL_RUN_SCRIPT" "base" "0" "$WHICH"; then
    last=0; echo "$last" > "$STATE_FILE"
  fi
fi

latest_ckpt_step() {
  ls -d "$TRAIN_OUTPUT_DIR"/checkpoint-* 2>/dev/null \
    | sed 's#.*/checkpoint-##' | grep -E '^[0-9]+$' | sort -n | tail -1
}

while true; do
  step=$(latest_ckpt_step)
  if [ -n "$step" ] && [ "$step" -gt "$last" ]; then
    ckpt="$TRAIN_OUTPUT_DIR/checkpoint-$step"
    echo "[watch] new checkpoint: $ckpt (skipping any between $last and $step)"
    START=$(date +%s)
    bash "$REPO/scripts/eval_live/$EVAL_RUN_SCRIPT" "$ckpt" "$step" "$WHICH"
    rc=$?
    echo "[watch] eval round for step=$step finished rc=$rc in $(( $(date +%s) - START ))s"
    if [ $rc -eq 0 ]; then last=$step; echo "$last" > "$STATE_FILE"; fi
  else
    sleep "$POLL_INTERVAL"
  fi
done

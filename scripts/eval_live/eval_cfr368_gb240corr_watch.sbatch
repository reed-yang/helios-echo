#!/bin/bash
#SBATCH --job-name=helios_e240c
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=mc-node02
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/e240c_%A_%a.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/e240c_%A_%a.out
# LIVE event-switch eval for the gb240 CONTINUE-with-corruptions run (train on mc-node01,c-node03-06;
# this eval uses 4 GPUs on mc-node02 only, leaving the other 4 free and freeing c-node08 entirely
# (low render volume: 5 cases per checkpoint, checkpoints land ~every 6h). Watches stage1_init_cfr368_gb240 for
# NEW checkpoints (>=4000: 4000 = pre-corruption baseline, then 4500..15000 every 500) and renders the
# 5 qb event-switch cases per checkpoint at 368x640/1386f with the MULTI-EVENT PROMPT-SWITCHING spec
# (identical params to eval_cfr368_correct_watch.sbatch, so directly comparable across runs).
# 16 array workers share an atomic-mv queue (post-outage lesson: job arrays, NOT multi-task srun; also
# HF_ENABLE_PARALLEL_LOADING=no). PERSISTENT: polls every 5 min until step-15000 has all 5 videos.
# Idempotent; stale run/ claims (>5h) are requeued.
# Launch:  sbatch scripts/eval_live/eval_cfr368_gb240corr_watch.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
TRAIN=/mnt/beegfs/xiangbo/helios_runs/stage1_init_cfr368_gb240
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb
OUTROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368_gb240corr   # -> $OUTROOT/evtsw/step-<N>/<id>.mp4
QDIR=$OUTROOT/.q240c
MIN_STEP=4000; LAST_STEP=15000
IDS="1000 1001 1002 1003 1010"
H=368; W=640; NF=1386; STEPS=50; GUID=5.0
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"

export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
export HF_ENABLE_PARALLEL_LOADING=no
ME=$(hostname)-a${SLURM_ARRAY_TASK_ID}
export TRITON_CACHE_DIR=/tmp/triton_${USER}_e240c_${ME} TMPDIR=/tmp/tmp_${USER}_e240c_${ME}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd "$REPO"
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 12 ))
echo "[$ME] $(date) worker start; watching $TRAIN for ckpts $MIN_STEP..$LAST_STEP; ids=$IDS"

all_done() {  # true when step-LAST_STEP has all 5 videos
  local n=0
  for id in $IDS; do [ -f "$OUTROOT/evtsw/step-$LAST_STEP/$id.mp4" ] && n=$((n+1)); done
  [ "$n" -eq 5 ]
}

while true; do
  # requeue stale claims (worker died mid-render); claim files are ${ME}__${step}__${id}
  shopt -s nullglob
  for f in "$QDIR"/run/*; do
    if [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 18000 ]; then
      bn=$(basename "$f"); t=${bn#*__}
      mv "$f" "$QDIR/pending/$t" 2>/dev/null && echo "[$ME] $(date) requeued stale $t"
    fi
  done
  # enqueue work: numeric checkpoints >= MIN_STEP whose weights are fully written (>=60s old)
  for d in "$TRAIN"/checkpoint-*; do
    [ -d "$d" ] || continue
    step=$(basename "$d"); step=${step#checkpoint-}
    [[ "$step" =~ ^[0-9]+$ ]] || continue          # skips checkpoint-15000-final etc.
    [ "$step" -ge "$MIN_STEP" ] || continue
    [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    [ -f "$d/transformer_partial.pth" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y "$d/transformer_partial.pth") )) -ge 60 ] || continue
    for id in $IDS; do
      [ -f "$OUTROOT/evtsw/step-$step/$id.mp4" ] && continue
      t="${step}__${id}"
      ls "$QDIR/run/" 2>/dev/null | grep -q "__${t}$" && continue
      [ -e "$QDIR/done/$t" ] && continue
      [ -e "$QDIR/pending/$t" ] || touch "$QDIR/pending/$t" 2>/dev/null
    done
  done
  # claim one task atomically
  claimed=""; task=""
  for f in "$QDIR"/pending/*; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    if mv "$f" "$QDIR/run/${ME}__${bn}" 2>/dev/null; then claimed="$QDIR/run/${ME}__${bn}"; task="$bn"; break; fi
  done
  shopt -u nullglob
  if [ -z "$task" ]; then
    if all_done; then echo "[$ME] $(date) step-$LAST_STEP complete -> exit"; break; fi
    sleep 300; continue
  fi
  step=${task%%__*}; id=${task#*__}
  ck="$TRAIN/checkpoint-$step"; out="$OUTROOT/evtsw/step-$step"; mkdir -p "$out"
  echo "[$ME] $(date) START step=$step id=$id"
  python infer_helios.py --base_model_path "$EVAL_BASE" --transformer_path "$EVAL_BASE" \
    --lora_path "$ck" --partial_path "$ck/transformer_partial.pth" \
    --interactive_prompt_csv_path "$QB/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
    --num_frames "$NF" --num_inference_steps "$STEPS" --guidance_scale "$GUID" \
    --height "$H" --width "$W" --fps 24 --output_folder "$out"
  if [ -f "$out/$id.mp4" ]; then
    mv -f "$claimed" "$QDIR/done/$task"; echo "[$ME] $(date) DONE $task"
  else
    mv -f "$claimed" "$QDIR/failed/${ME}__${task}.$(date +%s)"; echo "[$ME] $(date) FAILED $task"
  fi
done
echo "[$ME] $(date) worker exit"

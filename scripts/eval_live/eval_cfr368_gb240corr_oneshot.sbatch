#!/bin/bash
#SBATCH --job-name=helios_e240os
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=mc-node01,c-node03,c-node04
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/e240os_%A_%a.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/e240os_%A_%a.out
# ONE-SHOT event-switch render for the gb240 CONTINUE run's NEW checkpoints, using the freed
# training nodes (mc-node01,c-node03,c-node04 = 24 single-GPU workers). Renders the 5 qb cases per
# numeric checkpoint >= MIN_STEP that lacks output, then EXITS when the backlog is clear (does NOT
# poll forever, since training is paused — frees the nodes when done). Same multi-event
# prompt-SWITCHING spec/params as eval_cfr368_gb240corr_watch.sbatch (directly comparable).
# 24 array workers share an atomic-mv queue. Idempotent (skips checkpoints that already have .mp4).
# Launch:  sbatch scripts/eval_live/eval_cfr368_gb240corr_oneshot.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
TRAIN=/mnt/beegfs/xiangbo/helios_runs/stage1_init_cfr368_gb240
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb
OUTROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368_gb240corr   # -> $OUTROOT/evtsw/step-<N>/<id>.mp4
QDIR=$OUTROOT/.q240os
MIN_STEP=6500        # render the partial step-6500 remainder + all new-data checkpoints (7000..10500)
IDS="1000 1001 1002 1003 1010"
H=368; W=640; NF=1386; STEPS=50; GUID=5.0
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"

export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
export HF_ENABLE_PARALLEL_LOADING=no
ME=$(hostname)-a${SLURM_ARRAY_TASK_ID}
export TRITON_CACHE_DIR=/tmp/triton_${USER}_e240os_${ME} TMPDIR=/tmp/tmp_${USER}_e240os_${ME}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd "$REPO"
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 8 ))
echo "[$ME] $(date) one-shot worker start; TRAIN=$TRAIN MIN_STEP=$MIN_STEP ids=$IDS"

empty_dir() { ! ls -A "$1" >/dev/null 2>&1 || [ -z "$(ls -A "$1" 2>/dev/null)" ]; }

idle=0
while true; do
  shopt -s nullglob
  # enqueue: numeric checkpoints >= MIN_STEP, weights fully written (>=60s old), missing output
  for d in "$TRAIN"/checkpoint-*; do
    [ -d "$d" ] || continue
    step=$(basename "$d"); step=${step#checkpoint-}
    [[ "$step" =~ ^[0-9]+$ ]] || continue
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
    # nothing to claim: if pending AND run are both empty, all work is done -> exit (grace x2)
    if empty_dir "$QDIR/pending" && empty_dir "$QDIR/run"; then
      idle=$((idle+1))
      if [ "$idle" -ge 2 ]; then echo "[$ME] $(date) backlog clear -> exit"; break; fi
    else
      idle=0
    fi
    sleep 20; continue
  fi
  idle=0
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

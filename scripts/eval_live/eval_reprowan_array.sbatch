#!/bin/bash
#SBATCH --job-name=helios_rwarr
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=c-node03,c-node04   # inference nodes (training is on c05/06)
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G                     # cap per-task RAM (default = whole node -> only 1 task/node would fit)
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/rwarr_%A_%a.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/rwarr_%A_%a.out
# LIVE eval watcher (JOB-ARRAY variant) for the repro-wan (Stage-1-from-raw-Wan) run.
# Replaces eval_reprowan_watch.sbatch: post-outage, a multi-task `srun` STEP inside one sbatch allocation
# stalls at launch (zero worker output) on the recovered nodes, while a fresh single-task allocation loads
# fine. So each worker is its OWN single-GPU job-array task (the proven pattern) -- no inner multi-task srun.
# 8 persistent workers share an atomic-mv queue on beegfs; each loops forever: rescan checkpoint-*, enqueue
# missing (step,id) (idempotent), claim one via atomic mv, run it. Base = eval_base_init (transformer_init:
# Wan-valued base + materialized memory patches) + lora + partial. Skips any <step>/<id>.mp4 already present.
# HF parallel loading OFF (1 loader proc/worker) to avoid swamping the still-recovering filesystem.
# Launch:  sbatch scripts/eval_live/eval_reprowan_array.sbatch   (cancel: scancel <jobid>)
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
SN=/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init   # transformer_init base + materialized patches
TRAIN=/mnt/beegfs/xiangbo/helios_runs/stage1_repro_wan
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb     # per-id 6-tag CSVs 1000..1010
OUTROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr       # -> $OUTROOT/reprowan_step<N>/<id>.mp4
QDIR=$OUTROOT/.rwwatch_queue
IDS="1000 1001 1002 1003 1010"
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"

export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
export HF_ENABLE_PARALLEL_LOADING=no   # 1 loader proc/worker -> no beegfs thundering-herd
ME=$(hostname)-a${SLURM_ARRAY_TASK_ID}
export TRITON_CACHE_DIR=/tmp/triton_${USER}_rwarr_${ME} TMPDIR=/tmp/tmp_${USER}_rwarr_${ME}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd "$REPO"
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 20 ))   # stagger first loads (0..140s) across the 8 workers
echo "[$ME] $(date) worker start; base=$EVAL_BASE train=$TRAIN"

while true; do
  # (idempotent) enqueue any (step,id) whose mp4 is missing
  for d in "$TRAIN"/checkpoint-*; do
    [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    [ -f "$d/transformer_partial.pth" ] || continue
    step=$(basename "$d" | sed "s/checkpoint-//"); [[ "$step" =~ ^[0-9]+$ ]] || continue
    for id in $IDS; do
      [ -f "$OUTROOT/reprowan_step$step/$id.mp4" ] && continue
      t="${step}__${id}"
      [ -e "$QDIR/run/"*"__$t" ] 2>/dev/null && continue
      [ -e "$QDIR/done/$t" ] && continue
      touch "$QDIR/pending/$t" 2>/dev/null
    done
  done
  # claim one task
  claimed=""; task=""
  shopt -s nullglob
  for f in "$QDIR"/pending/*; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    if mv "$f" "$QDIR/run/${ME}__${bn}" 2>/dev/null; then claimed="$QDIR/run/${ME}__${bn}"; task="$bn"; break; fi
  done
  shopt -u nullglob
  if [ -z "$task" ]; then sleep 60; continue; fi    # no work -> wait for new checkpoints (persistent)
  step=${task%%__*}; id=${task#*__}
  ck="$TRAIN/checkpoint-$step"; out="$OUTROOT/reprowan_step$step"; mkdir -p "$out"
  echo "[$ME] $(date) START step=$step id=$id"
  python infer_helios.py --base_model_path "$SN" --transformer_path "$EVAL_BASE" \
    --lora_path "$ck" --partial_path "$ck/transformer_partial.pth" \
    --interactive_prompt_csv_path "$QB/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
    --num_frames 1386 --num_inference_steps 50 --guidance_scale 5.0 --height 384 --width 640 --fps 24 \
    --output_folder "$out"
  if [ -f "$out/$id.mp4" ]; then mv -f "$claimed" "$QDIR/done/$task"; echo "[$ME] $(date) DONE $task";
  else mv -f "$claimed" "$QDIR/failed/$task" 2>/dev/null; echo "[$ME] $(date) FAIL $task -> failed/"; fi
done

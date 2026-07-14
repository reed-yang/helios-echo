#!/bin/bash
#SBATCH --job-name=helios_rw20
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=c-node07,c-node08           # idle c07 (+ c08 as gb240corr frees)
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/rw20_%A_%a.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/rw20_%A_%a.out
# 20-step event-switch eval for the REWEIGHT_v1 run (stage1_lora_reweight_368).
# 56 ids (qb_long_rewrite) x 2 checkpoints {19500,27500} = 112 videos at 368x640, 1386f, event-switch.
# 16 single-GPU workers on mc-node02/c-node08, atomic (step x id) queue, NON-PERSISTENT (backfill then exit).
# Idempotent (skips existing step-<N>/<id>.mp4). Launch: sbatch scripts/eval_live/eval_reweight_qblong_20.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
TRAIN=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_reweight_368
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb_long_rewrite
OUTROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368/reweight_qb_long_20step
QDIR=$OUTROOT/.q
STEPS="19500 27500"
H=368; W=640; NF=1386; INFSTEPS=20; GUID=5.0                                      # <-- 20 inference steps
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"
IDS=$(ls "$QB"/*.csv 2>/dev/null | xargs -n1 basename | sed 's/\.csv$//' | sort)

export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL HF_ENABLE_PARALLEL_LOADING=no
ME=$(hostname)-a${SLURM_ARRAY_TASK_ID}
export TRITON_CACHE_DIR=/tmp/triton_${USER}_rw20_${ME} TMPDIR=/tmp/tmp_${USER}_rw20_${ME}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd "$REPO"
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 12 ))
echo "[$ME] $(date) worker start; steps=$STEPS ids=$(echo $IDS|wc -w) res=${H}x${W} infsteps=$INFSTEPS"

while true; do
  for step in $STEPS; do
    d="$TRAIN/checkpoint-$step"
    [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    [ -f "$d/transformer_partial.pth" ] || continue
    for id in $IDS; do
      [ -f "$OUTROOT/step-$step/$id.mp4" ] && continue
      t="${step}__${id}"
      [ -e "$QDIR/run/"*"__$t" ] 2>/dev/null && continue
      [ -e "$QDIR/done/$t" ] && continue
      touch "$QDIR/pending/$t" 2>/dev/null
    done
  done
  claimed=""; task=""
  shopt -s nullglob
  for f in "$QDIR"/pending/*; do
    [ -e "$f" ] || continue; bn=$(basename "$f")
    if mv "$f" "$QDIR/run/${ME}__${bn}" 2>/dev/null; then claimed="$QDIR/run/${ME}__${bn}"; task="$bn"; break; fi
  done
  shopt -u nullglob
  if [ -z "$task" ]; then echo "[$ME] $(date) no pending work -> exit"; break; fi
  step=${task%%__*}; id=${task#*__}
  ck="$TRAIN/checkpoint-$step"; out="$OUTROOT/step-$step"; mkdir -p "$out"
  echo "[$ME] $(date) START step=$step id=$id"
  python infer_helios.py --base_model_path "$EVAL_BASE" --transformer_path "$EVAL_BASE" \
    --lora_path "$ck" --partial_path "$ck/transformer_partial.pth" \
    --interactive_prompt_csv_path "$QB/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
    --num_frames "$NF" --num_inference_steps "$INFSTEPS" --guidance_scale "$GUID" \
    --height "$H" --width "$W" --fps 24 --output_folder "$out"
  if [ -f "$out/$id.mp4" ]; then mv -f "$claimed" "$QDIR/done/$task"; echo "[$ME] $(date) DONE $task";
  else mv -f "$claimed" "$QDIR/failed/$task" 2>/dev/null; echo "[$ME] $(date) FAIL $task"; fi
done

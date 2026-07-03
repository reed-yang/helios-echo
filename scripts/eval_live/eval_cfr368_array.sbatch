#!/bin/bash
#SBATCH --job-name=helios_eval368a
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=c-node07,c-node08          # the 2 idle eval nodes (training is on mc01/02,c03-06)
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G                            # cap per-task RAM (else each task grabs whole node -> 1/node)
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/eval368a_%A_%a.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/eval368a_%A_%a.out
# LIVE event-switch eval (JOB-ARRAY / FULL 16-GPU work-queue) for the stage1_lora_cfr_368 run.
# 16 independent single-GPU workers (8 per node across c07/c08) share an atomic-mv queue and pull
# (checkpoint x id) tasks -> NO idle GPUs (vs the per-node script that left 3/8 idle with 5 ids on 8 GPUs).
# Each task renders ONE 1-minute event-switch video (qb/<id>.csv, 6 events x interpolate_time 7 = 1386f) at
# 368x640 from checkpoint-<step> (lora + transformer_partial.pth) on eval_base_init. world_size=1 (no NCCL).
# Idempotent: skips any step-<N>/<id>.mp4 already present. Persistent: backfills then waits for new ckpts.
# HF parallel loading OFF (1 loader/worker) to avoid swamping beegfs with 16 concurrent model loads.
# Launch:  sbatch scripts/eval_live/eval_cfr368_array.sbatch     (cancel: scancel <jobid>)
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init      # base + transformer (materialized memory patches)
TRAIN=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb        # per-id 6-event CSVs 1000..1010
OUTROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368       # -> $OUTROOT/evtsw/step-<N>/<id>.mp4
QDIR=$OUTROOT/.q368
IDS="1000 1001 1002 1003 1010"
H=368; W=640; NF=1386; STEPS=50; GUID=5.0
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"

export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
export HF_ENABLE_PARALLEL_LOADING=no
ME=$(hostname)-a${SLURM_ARRAY_TASK_ID}
export TRITON_CACHE_DIR=/tmp/triton_${USER}_e368_${ME} TMPDIR=/tmp/tmp_${USER}_e368_${ME}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd "$REPO"
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 12 ))   # stagger first model loads (0..180s) across the 16 workers
echo "[$ME] $(date) worker start; base=$EVAL_BASE train=$TRAIN res=${H}x${W}"

while true; do
  # (idempotent) enqueue any (step,id) whose mp4 is missing
  for d in "$TRAIN"/checkpoint-*; do
    [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    [ -f "$d/transformer_partial.pth" ] || continue
    step=$(basename "$d" | sed "s/checkpoint-//"); [[ "$step" =~ ^[0-9]+$ ]] || continue
    for id in $IDS; do
      [ -f "$OUTROOT/evtsw/step-$step/$id.mp4" ] && continue
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
  # NON-PERSISTENT: when no pending task remains, EXIT and release the node (don't monopolize c07/c08 vs
  # siyuan's chain). The monitor cron resubmits to backfill new checkpoints. Idempotent across resubmits.
  if [ -z "$task" ]; then echo "[$ME] $(date) no pending work -> exit (release node)"; break; fi
  step=${task%%__*}; id=${task#*__}
  ck="$TRAIN/checkpoint-$step"; out="$OUTROOT/evtsw/step-$step"; mkdir -p "$out"
  echo "[$ME] $(date) START step=$step id=$id"
  python infer_helios.py --base_model_path "$EVAL_BASE" --transformer_path "$EVAL_BASE" \
    --lora_path "$ck" --partial_path "$ck/transformer_partial.pth" \
    --interactive_prompt_csv_path "$QB/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
    --num_frames "$NF" --num_inference_steps "$STEPS" --guidance_scale "$GUID" \
    --height "$H" --width "$W" --fps 24 --output_folder "$out"
  if [ -f "$out/$id.mp4" ]; then mv -f "$claimed" "$QDIR/done/$task"; echo "[$ME] $(date) DONE $task";
  else mv -f "$claimed" "$QDIR/failed/$task" 2>/dev/null; echo "[$ME] $(date) FAIL $task -> failed/"; fi
done

#!/bin/bash
#SBATCH --job-name=helios_basepar
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=c-node05,c-node06,c-node07,c-node08
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/basepar_%A_%a.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/basepar_%A_%a.out
# BASELINE (released Helios-Base, NO LoRA / NO partial) event-switch streaming inference over the 56
# PARAGRAPH-rewritten prompts (qb_long_paragraph/*.csv) for a fair base-vs-ours comparison. Same streaming
# params as our evtsw eval: 1386f, interpolate_time 7, 50 steps, 368x640, guidance 5.0. transformer_init
# already contains patch_short/mid/long, so no partial needed. 32 single-GPU workers (8/node x c05-c08)
# share an atomic-mv queue. NON-PERSISTENT: backfill then exit. Idempotent (skips existing mp4).
# Launch:  sbatch scripts/eval_live/eval_base_paragraph.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init      # released Helios-Base (base+transformer_init)
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb_long_paragraph   # 56 paragraph CSVs
OUTROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368/base_paragraph  # -> $OUTROOT/<id>.mp4
QDIR=$OUTROOT/.q
H=368; W=640; NF=1386; INFSTEPS=50; GUID=5.0
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed" "$OUTROOT"
IDS=$(ls "$QB"/*.csv 2>/dev/null | xargs -n1 basename | sed 's/\.csv$//' | sort)

export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL HF_ENABLE_PARALLEL_LOADING=no
ME=$(hostname)-a${SLURM_ARRAY_TASK_ID}
export TRITON_CACHE_DIR=/tmp/triton_${USER}_basepar_${ME} TMPDIR=/tmp/tmp_${USER}_basepar_${ME}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd "$REPO"
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 6 ))
echo "[$ME] $(date) BASELINE worker start; ids=$(echo $IDS|wc -w) res=${H}x${W} nf=$NF steps=$INFSTEPS"

while true; do
  for id in $IDS; do
    [ -f "$OUTROOT/$id.mp4" ] && continue
    [ -e "$QDIR/run/"*"__$id" ] 2>/dev/null && continue
    [ -e "$QDIR/done/$id" ] && continue
    touch "$QDIR/pending/$id" 2>/dev/null
  done
  claimed=""; task=""
  shopt -s nullglob
  for f in "$QDIR"/pending/*; do
    [ -e "$f" ] || continue; bn=$(basename "$f")
    if mv "$f" "$QDIR/run/${ME}__${bn}" 2>/dev/null; then claimed="$QDIR/run/${ME}__${bn}"; task="$bn"; break; fi
  done
  shopt -u nullglob
  if [ -z "$task" ]; then echo "[$ME] $(date) no pending work -> exit"; break; fi
  id="$task"
  echo "[$ME] $(date) START base id=$id"
  python infer_helios.py --base_model_path "$EVAL_BASE" --transformer_path "$EVAL_BASE" \
    --interactive_prompt_csv_path "$QB/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
    --num_frames "$NF" --num_inference_steps "$INFSTEPS" --guidance_scale "$GUID" \
    --height "$H" --width "$W" --fps 24 --output_folder "$OUTROOT"
  if [ -f "$OUTROOT/$id.mp4" ]; then mv -f "$claimed" "$QDIR/done/$task"; echo "[$ME] $(date) DONE $id";
  else mv -f "$claimed" "$QDIR/failed/$task" 2>/dev/null; echo "[$ME] $(date) FAIL $id"; fi
done

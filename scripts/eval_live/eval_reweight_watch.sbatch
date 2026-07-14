#!/bin/bash
#SBATCH --job-name=helios_erw
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=mc-node02
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/erw_%A_%a.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/erw_%A_%a.out
# LIVE checkpoint eval for the REWEIGHT run (train job on mc-node01,c-node03..07; eval owns
# mc-node02 + c-node08). Watches stage1_lora_reweight_368 for new checkpoints (>=20000, every 500,
# numeric AND the final checkpoint-29200-final) and renders the 14 demo scenes, v3 prompt version
# (eval_prompts_demo14_aligned/scene_*.csv, 6 prompts each -> hard prompt-switch every 7 chunks) per
# checkpoint at 368x640/1386f — same params as eval_cfr368_correct_watch.sbatch.
# 8 single-GPU array workers per node share an atomic-mv queue (job arrays, NOT multi-task srun;
# HF_ENABLE_PARALLEL_LOADING=no — post-outage lesson). PERSISTENT: polls every 5 min until the
# final step has all 14 videos, then exits. Idempotent; stale run/ claims (>5h) are requeued.
# Launch BOTH nodes (same queue dir, CLI nodelist overrides the directive):
#   sbatch scripts/eval_live/eval_reweight_watch.sbatch
#   sbatch --nodelist=c-node08 scripts/eval_live/eval_reweight_watch.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
TRAIN=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_reweight_368
PROMPTS=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_demo14_v3
OUTROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_reweight   # -> $OUTROOT/demo14_v3/step-<N>/<scene>.mp4
QDIR=$OUTROOT/.qerwv3
MIN_STEP=19500; LAST_STEP=31140   # 19500 = pre-finetune baseline (seeded symlink ckpt); 31140 = -final
IDS="scene_01 scene_02 scene_03 scene_03a scene_03b scene_03c scene_04 scene_05 scene_06 scene_07 scene_08 scene_09 scene_10 scene_11"
NIDS=14
H=368; W=640; NF=1386; STEPS=20; GUID=5.0   # STEPS 50->20 (user request 2026-07-06)
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"

export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
export HF_ENABLE_PARALLEL_LOADING=no
ME=$(hostname)-a${SLURM_ARRAY_TASK_ID}
export TRITON_CACHE_DIR=/tmp/triton_${USER}_erw_${ME} TMPDIR=/tmp/tmp_${USER}_erw_${ME}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd "$REPO"
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 12 ))
echo "[$ME] $(date) worker start; watching $TRAIN for ckpts $MIN_STEP..$LAST_STEP; $NIDS scenes"

ckpt_dir_for_step() {  # numeric step -> existing checkpoint dir (plain or -final)
  if [ -d "$TRAIN/checkpoint-$1" ]; then echo "$TRAIN/checkpoint-$1";
  elif [ -d "$TRAIN/checkpoint-$1-final" ]; then echo "$TRAIN/checkpoint-$1-final"; fi
}

all_done() {  # true when step-LAST_STEP has all videos
  local n=0
  for id in $IDS; do [ -f "$OUTROOT/demo14_v3/step-$LAST_STEP/$id.mp4" ] && n=$((n+1)); done
  [ "$n" -eq "$NIDS" ]
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
  # enqueue work: ONLY the 19500 baseline and the NEWEST checkpoint (user request 2026-07-06:
  # spot-check the freshest ckpt instead of backfilling every 500 steps). "Newest" is re-evaluated
  # every poll, so it follows training; intermediate ckpts are skipped, ending at 31140-final.
  latest=$(for d in "$TRAIN"/checkpoint-*; do
             n=$(basename "$d"); n=${n#checkpoint-}
             [[ "$n" =~ ^([0-9]+)(-final)?$ ]] && echo "${BASH_REMATCH[1]}"
           done | sort -n | tail -1)
  # prune stale pending tasks: keep only {19500, latest} (a ckpt that WAS latest but got
  # superseded before rendering would otherwise accumulate into a full backfill sweep)
  for f in "$QDIR"/pending/*; do
    [ -e "$f" ] || continue
    ps=$(basename "$f"); ps=${ps%%__*}
    if [ "$ps" != "19500" ] && [ "$ps" != "$latest" ]; then rm -f "$f"; fi
  done
  for step in 19500 $latest; do
    d=$(ckpt_dir_for_step "$step")
    [ -n "$d" ] || continue
    [ "$step" -ge "$MIN_STEP" ] || continue
    [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    [ -f "$d/transformer_partial.pth" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y "$d/transformer_partial.pth") )) -ge 60 ] || continue
    for id in $IDS; do
      [ -f "$OUTROOT/demo14_v3/step-$step/$id.mp4" ] && continue
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
  ck=$(ckpt_dir_for_step "$step")
  if [ -z "$ck" ]; then mv -f "$claimed" "$QDIR/failed/${ME}__${task}.$(date +%s)"; continue; fi
  out="$OUTROOT/demo14_v3/step-$step"; mkdir -p "$out"
  echo "[$ME] $(date) START step=$step id=$id ck=$ck"
  python infer_helios.py --base_model_path "$EVAL_BASE" --transformer_path "$EVAL_BASE" \
    --lora_path "$ck" --partial_path "$ck/transformer_partial.pth" \
    --interactive_prompt_csv_path "$PROMPTS/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
    --num_frames "$NF" --num_inference_steps "$STEPS" --guidance_scale "$GUID" \
    --height "$H" --width "$W" --fps 24 --output_folder "$out"
  if [ -f "$out/$id.mp4" ]; then
    mv -f "$claimed" "$QDIR/done/$task"; echo "[$ME] $(date) DONE $task"
  else
    mv -f "$claimed" "$QDIR/failed/${ME}__${task}.$(date +%s)"; echo "[$ME] $(date) FAILED $task"
  fi
done
echo "[$ME] $(date) worker exit"

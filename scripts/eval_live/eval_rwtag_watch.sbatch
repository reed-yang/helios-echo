#!/bin/bash
#SBATCH --job-name=helios_erwt
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=mc-node02
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/erwt_%A_%a.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/erwt_%A_%a.out
# MERGED live checkpoint eval: replaces eval_reweight_watch (jobs 4847/4848) so mc-node02+c-node08
# can serve BOTH training runs at once:
#   r1  = v1 reweight  (stage1_lora_reweight_368, ckpts 19500..31140) -> demo14 untagged
#         (same OUTROOT as before: eval_out_reweight/demo14 - already-rendered steps are skipped)
#   r2  = v2 rwtag  (stage1_lora_rwtag_368) -> eval_prompts_demo14_V3 (aligned + event-ID-free,
#         the recommended set), 14 scenes
#   r2t = v2 rwtag  -> eval_prompts_demo14_v3_TAG (9 dataset-mapped scenes; 05/08/09/10/11
#         have no matching training set and are skipped)
# r2/r2t are LATEST-ONLY (user request 2026-07-06): only the newest ready v2 checkpoint is
# enqueued; pending tasks for superseded steps are dropped (in-flight renders finish). When
# checkpoint-20784-final lands it becomes the latest and gets fully rendered. v2 tasks are
# claimed BEFORE the (large) r1 backlog. Everything else as eval_reweight_watch: 8 single-GPU
# array workers/node, atomic-mv queue, 5-min poll, stale-claim requeue, idempotent.
# Launch BOTH nodes (same queue dir; CLI nodelist overrides the directive):
#   sbatch scripts/eval_live/eval_rwtag_watch.sbatch
#   sbatch --nodelist=c-node08 scripts/eval_live/eval_rwtag_watch.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
TRAIN1=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_reweight_368
TRAIN2=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_rwtag_368
PROMPTS=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_demo14
PROMPTS_A=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_demo14_v3
PROMPTS_AT=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_demo14_v3_tag
OUT1=/mnt/beegfs/xiangbo/helios_runs/eval_out_reweight/demo14
OUT2=/mnt/beegfs/xiangbo/helios_runs/eval_out_rwtag/demo14_v3
OUT2T=/mnt/beegfs/xiangbo/helios_runs/eval_out_rwtag/demo14_v3_tag
QDIR=/mnt/beegfs/xiangbo/helios_runs/eval_out_rwtag/.qerwt
MIN1=19500; LAST1=31140    # r1: 19500 baseline .. 31140-final
MIN2=19600; LAST2=20784    # r2/r2t: latest-only, floor 19600, final 20784
IDS="scene_01 scene_02 scene_03 scene_03a scene_03b scene_03c scene_04 scene_05 scene_06 scene_07 scene_08 scene_09 scene_10 scene_11"
IDS_TAG="scene_01 scene_02 scene_03 scene_03a scene_03b scene_03c scene_04 scene_06 scene_07"
N1=14; N2=14; N2T=9
H=368; W=640; NF=1386; STEPS=50; STEPS_V2=20; GUID=5.0   # r1 stays 50 (matches its existing outputs); v2 = 20 (user request)
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed" "$OUT2" "$OUT2T"

export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
export HF_ENABLE_PARALLEL_LOADING=no
ME=$(hostname)-a${SLURM_ARRAY_TASK_ID}
export TRITON_CACHE_DIR=/tmp/triton_${USER}_erwt_${ME} TMPDIR=/tmp/tmp_${USER}_erwt_${ME}
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd "$REPO"
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 12 ))
echo "[$ME] $(date) worker start; r1=$TRAIN1($MIN1..$LAST1) r2=$TRAIN2($MIN2..$LAST2)"

ckpt_dir_for_step() {  # $1=train_dir $2=step -> existing checkpoint dir (plain or -final)
  if [ -d "$1/checkpoint-$2" ]; then echo "$1/checkpoint-$2";
  elif [ -d "$1/checkpoint-$2-final" ]; then echo "$1/checkpoint-$2-final"; fi
}

count_done() {  # $1=outdir $2=step $3=ids -> number of finished videos
  local n=0
  for id in $3; do [ -f "$1/step-$2/$id.mp4" ] && n=$((n+1)); done
  echo "$n"
}

all_done() {
  [ "$(count_done "$OUT1" "$LAST1" "$IDS")" -eq "$N1" ] || return 1
  [ "$(count_done "$OUT2" "$LAST2" "$IDS")" -eq "$N2" ] || return 1
  [ "$(count_done "$OUT2T" "$LAST2" "$IDS_TAG")" -eq "$N2T" ] || return 1
}

latest_ready_step() {  # $1=train_dir -> highest step whose weights are fully written ('' if none)
  local d name step best=-1
  for d in "$1"/checkpoint-*; do
    [ -d "$d" ] || continue
    name=$(basename "$d"); name=${name#checkpoint-}
    [[ "$name" =~ ^([0-9]+)(-final)?$ ]] || continue
    step=${BASH_REMATCH[1]}
    [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    [ -f "$d/transformer_partial.pth" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y "$d/transformer_partial.pth") )) -ge 60 ] || continue
    [ "$step" -gt "$best" ] && best=$step
  done
  [ "$best" -ge 0 ] && echo "$best"
}

enqueue_latest() {  # $1=runkey $2=train_dir $3=min $4=outroot $5=ids  (latest checkpoint only)
  local run=$1 train=$2 min=$3 outroot=$4 ids=$5 step id t f
  step=$(latest_ready_step "$train"); [ -n "$step" ] || return 0
  [ "$step" -ge "$min" ] || return 0
  # drop pending tasks of this run that target a superseded step
  for f in "$QDIR"/pending/${run}__*; do
    [ -e "$f" ] || continue
    [[ "$(basename "$f")" == "${run}__${step}__"* ]] || rm -f "$f" 2>/dev/null
  done
  for id in $ids; do
    [ -f "$outroot/step-$step/$id.mp4" ] && continue
    t="${run}__${step}__${id}"
    ls "$QDIR/run/" 2>/dev/null | grep -q "__${t}$" && continue
    [ -e "$QDIR/done/$t" ] && continue
    [ -e "$QDIR/pending/$t" ] || touch "$QDIR/pending/$t" 2>/dev/null
  done
}

enqueue_run() {  # $1=runkey(r1|r2|r2t) $2=train_dir $3=min $4=outroot $5=ids
  local run=$1 train=$2 min=$3 outroot=$4 ids=$5 d name step id t
  for d in "$train"/checkpoint-*; do
    [ -d "$d" ] || continue
    name=$(basename "$d"); name=${name#checkpoint-}
    [[ "$name" =~ ^([0-9]+)(-final)?$ ]] || continue
    step=${BASH_REMATCH[1]}
    [ "$step" -ge "$min" ] || continue
    [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    [ -f "$d/transformer_partial.pth" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y "$d/transformer_partial.pth") )) -ge 60 ] || continue
    for id in $ids; do
      [ -f "$outroot/step-$step/$id.mp4" ] && continue
      t="${run}__${step}__${id}"
      ls "$QDIR/run/" 2>/dev/null | grep -q "__${t}$" && continue
      [ -e "$QDIR/done/$t" ] && continue
      [ -e "$QDIR/pending/$t" ] || touch "$QDIR/pending/$t" 2>/dev/null
    done
  done
}

while true; do
  # requeue stale claims (worker died mid-render); claim files are ${ME}__${run}__${step}__${id}
  shopt -s nullglob
  for f in "$QDIR"/run/*; do
    if [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 18000 ]; then
      bn=$(basename "$f"); t=${bn#*__}
      mv "$f" "$QDIR/pending/$t" 2>/dev/null && echo "[$ME] $(date) requeued stale $t"
    fi
  done
  enqueue_run r1 "$TRAIN1" "$MIN1" "$OUT1" "$IDS"
  enqueue_latest r2 "$TRAIN2" "$MIN2" "$OUT2" "$IDS"
  enqueue_latest r2t "$TRAIN2" "$MIN2" "$OUT2T" "$IDS_TAG"
  # claim one task atomically - v2 (r2t/r2, small latest-only sets) BEFORE the big r1 backlog
  claimed=""; task=""
  for f in "$QDIR"/pending/r2t__* "$QDIR"/pending/r2__* "$QDIR"/pending/*; do
    [ -e "$f" ] || continue
    bn=$(basename "$f")
    if mv "$f" "$QDIR/run/${ME}__${bn}" 2>/dev/null; then claimed="$QDIR/run/${ME}__${bn}"; task="$bn"; break; fi
  done
  shopt -u nullglob
  if [ -z "$task" ]; then
    if all_done; then echo "[$ME] $(date) all three finals complete -> exit"; break; fi
    sleep 300; continue
  fi
  run=${task%%__*}; rest=${task#*__}; step=${rest%%__*}; id=${rest#*__}
  case "$run" in
    r1)  train=$TRAIN1; outroot=$OUT1; pdir=$PROMPTS;    nsteps=$STEPS ;;
    r2)  train=$TRAIN2; outroot=$OUT2; pdir=$PROMPTS_A;  nsteps=$STEPS_V2 ;;
    r2t) train=$TRAIN2; outroot=$OUT2T; pdir=$PROMPTS_AT; nsteps=$STEPS_V2 ;;
    *)   mv -f "$claimed" "$QDIR/failed/${ME}__${task}.$(date +%s)"; continue ;;
  esac
  ck=$(ckpt_dir_for_step "$train" "$step")
  if [ -z "$ck" ]; then mv -f "$claimed" "$QDIR/failed/${ME}__${task}.$(date +%s)"; continue; fi
  out="$outroot/step-$step"; mkdir -p "$out"
  echo "[$ME] $(date) START run=$run step=$step id=$id ck=$ck"
  python infer_helios.py --base_model_path "$EVAL_BASE" --transformer_path "$EVAL_BASE" \
    --lora_path "$ck" --partial_path "$ck/transformer_partial.pth" \
    --interactive_prompt_csv_path "$pdir/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
    --num_frames "$NF" --num_inference_steps "$nsteps" --guidance_scale "$GUID" \
    --height "$H" --width "$W" --fps 24 --output_folder "$out"
  if [ -f "$out/$id.mp4" ]; then
    mv -f "$claimed" "$QDIR/done/$task"; echo "[$ME] $(date) DONE $task"
  else
    mv -f "$claimed" "$QDIR/failed/${ME}__${task}.$(date +%s)"; echo "[$ME] $(date) FAILED $task"
  fi
done
echo "[$ME] $(date) worker exit"

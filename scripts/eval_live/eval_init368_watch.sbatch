#!/bin/bash
#SBATCH --job-name=helios_init368w
#SBATCH --nodes=1
#SBATCH --nodelist=c-node07            # dedicated inference node (idle; not in the 1-6 training set)
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=10
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/init368w_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/init368w_%j.out
# LIVE eval watcher for the full Stage-1-INIT (from-raw-Wan) run `stage1_init_cfr368` (job 4127). Renders the
# 5 one-minute event-switch cases (ids 1000/1001/1002/1003/1010; 6 events x interpolate_time 7 = 1386 frames)
# for EVERY saved checkpoint, at 368x640. 8 single-GPU workers (world_size=1, NO NCCL) on c-node07; each loops:
# rescan stage1_init_cfr368/checkpoint-*, enqueue missing (step,id), claim one via atomic mv, render. From-Wan
# run -> transformer base = eval_base_init (transformer_init: Wan-valued base + materialized memory patches),
# then lora + partial from the checkpoint (same wiring as eval_reprowan_*). Idempotent; persistent until scancel.
# Launch:  sbatch scripts/eval_live/eval_init368_watch.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
SN=/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init   # transformer_init base + materialized memory patches
TRAIN=/mnt/beegfs/xiangbo/helios_runs/stage1_init_cfr368
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb     # per-id 6-tag CSVs 1000..1010
OUTROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368    # -> $OUTROOT/init368_step<N>/<id>.mp4
QDIR=$OUTROOT/.init368w_queue
IDS="1000 1001 1002 1003 1010"
H=368; W=640; NF=1386; STEPS=50; GUID=5.0
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"
echo "[init368w] $(date) live eval watcher start on c-node07; base=$EVAL_BASE train=$TRAIN res=${H}x${W}"

srun --ntasks=8 --ntasks-per-node=8 --gres=gpu:h200:8 --cpus-per-task=10 --exact --kill-on-bad-exit=0 bash -c '
  set -uo pipefail
  export PATH='"$PY_ENV"'/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
  export HF_ENABLE_PARALLEL_LOADING=no
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  ME=$(hostname)-g$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_init368w_${ME} TMPDIR=/tmp/tmp_'"$USER"'_init368w_${ME}
  mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd '"$REPO"'
  sleep $(( ${SLURM_PROCID:-0} * 20 ))   # stagger model loads to avoid beegfs thundering-herd
  SN='"$SN"'; EVAL_BASE='"$EVAL_BASE"'; TRAIN='"$TRAIN"'; QB='"$QB"'; OUTROOT='"$OUTROOT"'; QDIR='"$QDIR"'
  IDS="'"$IDS"'"; H='"$H"'; W='"$W"'; NF='"$NF"'; STEPS='"$STEPS"'; GUID='"$GUID"'
  while true; do
    for d in "$TRAIN"/checkpoint-*; do
      [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
      [ -f "$d/transformer_partial.pth" ] || continue
      step=$(basename "$d" | sed "s/checkpoint-//"); [[ "$step" =~ ^[0-9]+$ ]] || continue
      for id in $IDS; do
        [ -f "$OUTROOT/init368_step$step/$id.mp4" ] && continue
        t="${step}__${id}"
        [ -e "$QDIR/run/"*"__$t" ] 2>/dev/null && continue
        [ -e "$QDIR/done/$t" ] && continue
        touch "$QDIR/pending/$t" 2>/dev/null
      done
    done
    claimed=""; task=""
    shopt -s nullglob
    for f in "$QDIR"/pending/*; do
      [ -e "$f" ] || continue
      bn=$(basename "$f")
      if mv "$f" "$QDIR/run/${ME}__${bn}" 2>/dev/null; then claimed="$QDIR/run/${ME}__${bn}"; task="$bn"; break; fi
    done
    shopt -u nullglob
    if [ -z "$task" ]; then sleep 60; continue; fi
    step=${task%%__*}; id=${task#*__}
    ck="$TRAIN/checkpoint-$step"; out="$OUTROOT/init368_step$step"; mkdir -p "$out"
    echo "[$ME] $(date) START step=$step id=$id"
    python infer_helios.py --base_model_path "$SN" --transformer_path "$EVAL_BASE" \
      --lora_path "$ck" --partial_path "$ck/transformer_partial.pth" \
      --interactive_prompt_csv_path "$QB/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
      --num_frames "$NF" --num_inference_steps "$STEPS" --guidance_scale "$GUID" \
      --height "$H" --width "$W" --fps 24 --output_folder "$out"
    if [ -f "$out/$id.mp4" ]; then mv -f "$claimed" "$QDIR/done/$task"; echo "[$ME] $(date) DONE $task";
    else mv -f "$claimed" "$QDIR/failed/$task" 2>/dev/null; echo "[$ME] $(date) FAIL $task"; fi
  done
'
echo "[init368w] $(date) watcher exited"

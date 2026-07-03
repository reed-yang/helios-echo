#!/bin/bash
#SBATCH --job-name=helios_evalfb368
#SBATCH --nodes=1
#SBATCH --nodelist=mc-node02                 # dedicated eval node (has 1 flaky GPU -> auto-quarantined below)
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=168:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/evalfb368_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/evalfb368_%j.out
# Live eval WATCHER for the full-base368 run. For each TARGET checkpoint step (250,1000,1750,...; i.e.
# 250 + 750k), once checkpoint-<step> is saved it: (1) CONSOLIDATES the DeepSpeed ZeRO-2 full-FT ckpt
# into an inference transformer (hybrid: fuse patch_embedding LoRA), then (2) renders the 56 qb_long_rewrite
# event-switch cases (6-segment, 1386f, 368x640, 50 steps) via 8 single-GPU workers sharing an atomic-mv
# queue. Bad GPU auto-quarantines after consecutive failures; failed tasks retry on other GPUs. Idempotent
# (skips existing mp4). Persistent (keeps watching as training produces new target checkpoints).
# Launch:  sbatch scripts/eval_live/eval_fullbase368_watch.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
MERGED=/mnt/beegfs/xiangbo/helios_runs/_merged/stage1_lora368_merged17000
TRAIN=/mnt/beegfs/xiangbo/helios_runs/stage1_fullbase_cfr368
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb_long_rewrite            # 56 per-id 6-segment CSVs
OUTROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368_fullbase/qb_long_rewrite   # -> step-<N>/<id>.mp4
CONS=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368_fullbase/consolidated         # -> step-<N>/transformer/
QDIR=$OUTROOT/.q
TARGET_STEPS="${TARGET_STEPS:-250 1000 1750 2500 3250 4000 4750 5500}"
NGPU=${NGPU:-8}; H=368; W=640; NF=1386; INFSTEPS=50; GUID=5.0; MAXRETRY=3; QUARANTINE=3
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$QDIR"/{pending,run,done,failed,retry} "$CONS"
IDS=$(ls "$QB"/*.csv 2>/dev/null | xargs -n1 basename | sed 's/\.csv$//' | sort)
NID=$(echo "$IDS" | wc -w)

export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL HF_ENABLE_PARALLEL_LOADING=no
cd "$REPO"
echo "[evalfb] $(date) start; target_steps={$TARGET_STEPS} ids=$NID ngpu=$NGPU res=${H}x${W}"

marker() { echo "$CONS/step-$1/transformer/diffusion_pytorch_model.safetensors.index.json"; }

consolidate() {  # $1=step ; CPU-only, ~few min. idempotent.
  local step=$1 ck="$TRAIN/checkpoint-$1"
  [ -f "$(marker "$step")" ] && return 0
  [ -d "$ck/pytorch_model" ] || return 1
  # skip if the checkpoint dir is still being written (mtime < 90s ago)
  local age=$(( $(date +%s) - $(stat -c %Y "$ck/pytorch_model" 2>/dev/null || echo 0) ))
  [ "$age" -lt 90 ] && { echo "[evalfb] step $step ckpt too fresh (${age}s) - wait"; return 1; }
  echo "[evalfb] $(date) CONSOLIDATE step $step -> $CONS/step-$step/transformer"
  mkdir -p "$CONS/step-$step"
  CUDA_VISIBLE_DEVICES="" python tools/offload_data/consolidate_fullbase_lorapatch.py \
    "$ck" "$MERGED" --subfolder transformer --out "$CONS/step-$step/transformer" \
    >> /mnt/beegfs/xiangbo/helios_runs/logs/evalfb368_consolidate.log 2>&1
  [ -f "$(marker "$step")" ]
}

worker() {  # $1=gpu index
  local g=$1 consec=0
  export CUDA_VISIBLE_DEVICES=$g
  export TRITON_CACHE_DIR=/tmp/triton_${USER}_efb_g$g TMPDIR=/tmp/tmp_${USER}_efb_g$g
  mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"
  while true; do
    local task="" f bn
    shopt -s nullglob
    for f in "$QDIR"/pending/*; do
      [ -e "$f" ] || continue; bn=$(basename "$f")
      if mv "$f" "$QDIR/run/g${g}__$bn" 2>/dev/null; then task=$bn; break; fi
    done
    shopt -u nullglob
    if [ -z "$task" ]; then sleep 15; continue; fi
    local step=${task%%__*} id=${task#*__} out="$OUTROOT/${task%%__*}"; mkdir -p "$OUTROOT/step-$step"
    out="$OUTROOT/step-$step"
    if [ -f "$out/$id.mp4" ]; then mv -f "$QDIR/run/g${g}__$task" "$QDIR/done/$task" 2>/dev/null; continue; fi
    echo "[efb g$g] $(date) START step=$step id=$id"
    python infer_helios.py --base_model_path "$EVAL_BASE" --transformer_path "$CONS/step-$step" \
      --interactive_prompt_csv_path "$QB/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
      --num_frames "$NF" --num_inference_steps "$INFSTEPS" --guidance_scale "$GUID" \
      --height "$H" --width "$W" --fps 24 --output_folder "$out" \
      >> "/mnt/beegfs/xiangbo/helios_runs/logs/evalfb368_infer_g$g.log" 2>&1
    if [ -f "$out/$id.mp4" ]; then
      mv -f "$QDIR/run/g${g}__$task" "$QDIR/done/$task" 2>/dev/null; consec=0
      echo "[efb g$g] $(date) DONE $task"
    else
      local r=$(cat "$QDIR/retry/$task" 2>/dev/null || echo 0); r=$((r+1)); echo "$r" > "$QDIR/retry/$task"
      consec=$((consec+1))
      if [ "$r" -ge "$MAXRETRY" ]; then mv -f "$QDIR/run/g${g}__$task" "$QDIR/failed/$task" 2>/dev/null
      else mv -f "$QDIR/run/g${g}__$task" "$QDIR/pending/$task" 2>/dev/null; fi
      echo "[efb g$g] $(date) FAIL $task (retry=$r consec=$consec)"
      if [ "$consec" -ge "$QUARANTINE" ]; then echo "[efb g$g] $(date) QUARANTINE gpu $g after $consec consecutive fails -> worker exit"; return 0; fi
    fi
  done
}

# launch GPU workers
for g in $(seq 0 $((NGPU-1))); do worker "$g" & done
echo "[evalfb] launched $NGPU workers"

# main enqueue/consolidate loop (persistent)
while true; do
  for step in $TARGET_STEPS; do
    [ -d "$TRAIN/checkpoint-$step" ] || continue
    if [ ! -f "$(marker "$step")" ]; then consolidate "$step" || continue; fi
    for id in $IDS; do
      [ -f "$OUTROOT/step-$step/$id.mp4" ] && continue
      t="${step}__${id}"
      [ -e "$QDIR/done/$t" ] && continue
      [ -e "$QDIR/failed/$t" ] && continue
      compgen -G "$QDIR/run/*__$t" >/dev/null 2>&1 && continue
      [ -e "$QDIR/pending/$t" ] && continue
      touch "$QDIR/pending/$t" 2>/dev/null
    done
  done
  sleep 120
done

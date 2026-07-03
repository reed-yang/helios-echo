#!/bin/bash
# Preview inference of a mid-training init/post checkpoint every 500 steps.
# Merges the LoRA checkpoint INLINE on the current (big-RAM) node (node7 + target
# inference nodes all run --mem=0 so have no free RAM), then submits a t2v inference
# job constrained to mc-node02 / c-node08 (picks whichever frees first).
#
#   bash scripts/wan22/preview_infer_step.sh <STEP> [CKPT_ROOT] [BASE] [LABEL_PREFIX]
#
# Idempotent: skips the merge if the merged dir exists, skips entirely if inference
# output already exists. Safe to call from the watchdog each tick.
set -euo pipefail
STEP=${1:?usage: preview_infer_step.sh STEP [CKPT_ROOT] [BASE] [LABEL_PREFIX]}
RUNS=/mnt/beegfs/xiangbo/helios_runs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
CKPT_ROOT=${2:-$RUNS/stage1_init_wan22full}
BASE=${3:-/mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers}
PREFIX=${4:-init}
PYBIN=/mnt/beegfs/yuheng/miniconda3/envs/helios/bin/python

CKPT=$CKPT_ROOT/checkpoint-$STEP
MERGED=$RUNS/preview_merged/${PREFIX}_step${STEP}
OUT=$RUNS/eval_out_wan22/${PREFIX}_step${STEP}
# Eval spec (multi-event prompt-SWITCHING): the 5 canonical qb cases in evtsw.csv, 6 events each,
# 1-minute videos (1386 frames). CSV + NUM_FRAMES + interpolate flags are defaulted in infer_wan22.sbatch.

if [ ! -d "$CKPT" ]; then echo "preview: no checkpoint $CKPT"; exit 0; fi
if [ -d "$OUT" ] && [ "$(ls -A "$OUT"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "preview: $OUT already has videos -> skip"; exit 0
fi
if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "prev_${PREFIX}_${STEP}"; then
  echo "preview: infer job prev_${PREFIX}_${STEP} already queued/running -> skip"; exit 0
fi

# 1) inline CPU merge (only if not already merged)
if [ ! -d "$MERGED/transformer" ]; then
  echo "preview: merging $CKPT -> $MERGED (inline CPU)"
  mkdir -p "$MERGED"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" $PYBIN scripts/wan22/merge_lora_wan22.py \
    --base "$BASE" --pipe_base /mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers \
    --lora "$CKPT/pytorch_lora_weights.safetensors" \
    --partial "$CKPT/transformer_partial.pth" \
    --out "$MERGED"
else
  echo "preview: merged dir exists -> reuse $MERGED"
fi

# 2) submit inference constrained to the two inference nodes (1 of the 2, whichever frees)
cd "$REPO"
JID=$(TRANSFORMER=$MERGED LABEL=${PREFIX}_step${STEP} OUT=$OUT \
  sbatch --parsable --nodes=1 --nodelist=mc-node02,c-node08 --gres=gpu:h200:8 \
  --job-name=prev_${PREFIX}_${STEP} scripts/wan22/infer_wan22.sbatch)
echo "preview: submitted infer job $JID (name=prev_${PREFIX}_${STEP}) -> $OUT"

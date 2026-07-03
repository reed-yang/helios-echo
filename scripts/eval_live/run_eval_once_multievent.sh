#!/bin/bash
# Generate multi-event (prompt-switching) eval videos from a trained LoRA checkpoint
# (data-parallel) and score them with HeliosBench reference-free metrics.
#
# Usage:  run_eval_once_multievent.sh <checkpoint_dir|base> <step> [test|train|both]
#
# Env knobs (defaults): REPO, PY_ENV, BASE_MODEL, EVAL_PROMPT_DIR, EVAL_OUT_DIR,
#   NUM_STEPS=50  GUIDANCE=5.0  H=384 W=640  NGPU=8  EVAL_GPUS  RUN_METRICS=1
# Video length is PER-VIDEO (from the spec), so there is no global NUM_FRAMES and no rename:
# infer_multievent.py writes {id}_{nf}_ori{nf}.mp4 directly.
set -euo pipefail

CKPT=${1:?checkpoint dir or 'base'}
STEP=${2:?step number}
WHICH=${3:-both}

REPO=${REPO:-/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team}
PY_ENV=${PY_ENV:-/mnt/beegfs/yuheng/miniconda3/envs/helios}
BASE_MODEL=${BASE_MODEL:-BestWishYsh/Helios-Base}
EVAL_PROMPT_DIR=${EVAL_PROMPT_DIR:-/mnt/beegfs/xiangbo/helios_runs/eval_prompts_multievent}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-/mnt/beegfs/xiangbo/helios_runs/eval_out_multievent}
NUM_STEPS=${NUM_STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}
H=${H:-384}; W=${W:-640}; NGPU=${NGPU:-8}
RUN_METRICS=${RUN_METRICS:-1}

export HF_HOME=${HF_HOME:-/mnt/beegfs/xiangbo/.cache/huggingface}
# do NOT set HF_HUB_OFFLINE: the flash-attn3 kernel is fetched at runtime (pre-cached).
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export PATH=$PY_ENV/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_${USER}_evalme_${SLURM_JOB_ID:-$$}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_evalme_${SLURM_JOB_ID:-$$}
# Node-local TMPDIR: Triton builds its launcher .c via gcc in $TMPDIR; an inherited beegfs
# TMPDIR + many concurrent compiles causes transient ENOENT on Python headers -> gcc fails.
export TMPDIR=/tmp/tmp_${USER}_evalme_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

gen_one() {  # $1 = set name (test/train)
  local set=$1
  local spec=$EVAL_PROMPT_DIR/eval_me_${set}.json
  local out=$EVAL_OUT_DIR/step-${STEP}/${set}
  mkdir -p "$out"
  echo "[gen] step=$STEP set=$set -> $out"
  cd "$REPO"
  local lora_args=(--lora_path "$CKPT")
  if [ "$STEP" = "0" ] || [ "$CKPT" = "base" ]; then lora_args=(); echo "[gen] BASELINE (no LoRA)"; fi
  # GPU set: EVAL_GPUS overrides (skip c-node08 GPU1, which thermally throttles and wedges its rank).
  local gpus=${EVAL_GPUS:-$(seq -s, 0 $((NGPU-1)))}
  local ngpu=$(echo "$gpus" | tr ',' '\n' | grep -c .)
  echo "[gen] using GPUs: $gpus ($ngpu procs)"
  CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=$ngpu --master_port=29612 \
    infer_multievent.py \
      --spec_json "$spec" \
      --base_model_path "$BASE_MODEL" \
      --transformer_path "$BASE_MODEL" \
      "${lora_args[@]}" \
      --num_inference_steps "$NUM_STEPS" --guidance_scale "$GUIDANCE" \
      --height "$H" --width "$W" \
      --output_folder "$out"
  cp "$EVAL_PROMPT_DIR/eval_me_meta_${set}.csv" "$out/_input.csv"
}

run_metrics() {  # $1 = set name
  local set=$1
  local vdir=$EVAL_OUT_DIR/step-${STEP}/${set}
  local odir=$EVAL_OUT_DIR/step-${STEP}/metrics_${set}
  local csv=$vdir/_input.csv
  mkdir -p "$odir"
  cd "$REPO/eval"
  echo "[metrics] step=$STEP set=$set -> $odir"
  CUDA_VISIBLE_DEVICES=0 python 0_get_aesthetic.py --input_csv "$csv" --video_dir "$vdir" --output_path "$odir" \
      --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
      --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth" &
  CUDA_VISIBLE_DEVICES=1 python 1_get_motion_amplitude.py --input_csv "$csv" --video_dir "$vdir" --output_path "$odir" --num_workers 16 &
  CUDA_VISIBLE_DEVICES=2 python 2_get_motion_smoothness.py --input_csv "$csv" --video_dir "$vdir" --output_path "$odir" \
      --smoothness_model_path "checkpoints/amt_model/amt-s.pth" &
  CUDA_VISIBLE_DEVICES=3 python 3_get_semantic.py --input_csv "$csv" --video_dir "$vdir" --output_path "$odir" \
      --semantic_model_path "checkpoints/ViCLIP" &
  CUDA_VISIBLE_DEVICES=4 python 5_get_drifting_aesthetic.py --input_csv "$csv" --video_dir "$vdir" --output_path "$odir" \
      --clip_model_path "checkpoints/aesthetic_model/ViT-L-14.pt" \
      --aesthetic_model_path "checkpoints/aesthetic_model/sa_0_4_vit_l_14_linear.pth" &
  CUDA_VISIBLE_DEVICES=5 python 6_get_drifting_motion_smoothness.py --input_csv "$csv" --video_dir "$vdir" --output_path "$odir" \
      --smoothness_model_path "checkpoints/amt_model/amt-s.pth" &
  CUDA_VISIBLE_DEVICES=6 python 7_get_drifting_semantic.py --input_csv "$csv" --video_dir "$vdir" --output_path "$odir" \
      --semantic_model_path "checkpoints/ViCLIP" &
  wait
  python 9_merge_all_scores.py --input_dir "$odir" --is_long || true
  echo "[metrics] done -> $odir"
}

SETS=()
case "$WHICH" in test) SETS=(test);; train) SETS=(train);; both) SETS=(test train);; esac
for s in "${SETS[@]}"; do gen_one "$s"; done
if [ "$RUN_METRICS" = "1" ]; then for s in "${SETS[@]}"; do run_metrics "$s"; done; fi
echo "[run_eval_once_multievent] step=$STEP complete."

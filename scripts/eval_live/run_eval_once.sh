#!/bin/bash
# Generate eval videos from a trained LoRA checkpoint (8-GPU data-parallel) and
# optionally score them with HeliosBench reference-free metrics.
#
# Usage:
#   run_eval_once.sh <checkpoint_dir> <step> [test|train|both]
#
# Env knobs (with defaults):
#   REPO, PY_ENV, BASE_MODEL, EVAL_PROMPT_DIR, EVAL_OUT_DIR,
#   NUM_FRAMES=330  NUM_STEPS=50  GUIDANCE=5.0  H=384 W=640  NGPU=8
#   RUN_METRICS=1   (0 to skip metrics; naturalness needs NATURALNESS_API_KEY)
set -euo pipefail

CKPT=${1:?checkpoint dir}
STEP=${2:?step number}
WHICH=${3:-both}

REPO=${REPO:-/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team}
PY_ENV=${PY_ENV:-/mnt/beegfs/yuheng/miniconda3/envs/helios}
BASE_MODEL=${BASE_MODEL:-BestWishYsh/Helios-Base}
EVAL_PROMPT_DIR=${EVAL_PROMPT_DIR:-/mnt/beegfs/xiangbo/helios_runs/eval_prompts}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-/mnt/beegfs/xiangbo/helios_runs/eval_out}
NUM_FRAMES=${NUM_FRAMES:-330}
NUM_STEPS=${NUM_STEPS:-50}
GUIDANCE=${GUIDANCE:-5.0}
H=${H:-384}; W=${W:-640}; NGPU=${NGPU:-8}
RUN_METRICS=${RUN_METRICS:-1}

export HF_HOME=${HF_HOME:-/mnt/beegfs/xiangbo/.cache/huggingface}
# do NOT set HF_HUB_OFFLINE: the flash-attn3 kernel is fetched at runtime (pre-cached).
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export PATH=$PY_ENV/bin:$PATH
# Node-local Triton/inductor cache: 8 concurrent compile procs race on a beegfs cache.
export TRITON_CACHE_DIR=/tmp/triton_${USER}_eval_${SLURM_JOB_ID:-$$}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_eval_${SLURM_JOB_ID:-$$}

gen_one() {  # $1 = set name (test/train)
  local set=$1
  local prompts=$EVAL_PROMPT_DIR/eval_prompts_${set}.txt
  local out=$EVAL_OUT_DIR/step-${STEP}/${set}
  mkdir -p "$out"
  echo "[gen] step=$STEP set=$set -> $out  ($(wc -l < "$prompts") prompts, ${NUM_FRAMES}f)"
  cd "$REPO"
  # Baseline (pre-fine-tune): step 0 or CKPT="base" -> evaluate raw Helios-Base, no adapter.
  local lora_args=(--lora_path "$CKPT")
  if [ "$STEP" = "0" ] || [ "$CKPT" = "base" ]; then lora_args=(); echo "[gen] BASELINE (no LoRA)"; fi
  local lora_str=""; [ ${#lora_args[@]} -gt 0 ] && lora_str="--lora_path $CKPT"
  if [ "${EVAL_NNODES:-1}" -gt 1 ]; then
    # MULTI-NODE data-parallel: 1 srun task per GPU across all allocated nodes. infer_helios
    # inits its process group from RANK/WORLD_SIZE env and shards prompts by global rank
    # (prompt_list[rank::world_size]) — pure data-parallel, ranks only sync at init. Each task
    # is pinned to its local GPU via CUDA_VISIBLE_DEVICES=$SLURM_LOCALID (device_count=1).
    local total=$(( EVAL_NNODES * 8 ))
    echo "[gen] MULTI-NODE: $EVAL_NNODES nodes x 8 = $total GPUs (data-parallel over $(wc -l < "$prompts") prompts)"
    srun --nodes="$EVAL_NNODES" --ntasks="$total" --ntasks-per-node=8 --gres=gpu:h200:8 \
         --kill-on-bad-exit=1 bash -c '
      export RANK=$SLURM_PROCID WORLD_SIZE='"$total"' LOCAL_RANK=$SLURM_LOCALID
      export MASTER_ADDR='"${EVAL_MASTER_ADDR}"' MASTER_PORT='"${EVAL_MASTER_PORT:-29615}"'
      export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
      export NCCL_DEBUG=WARN
      export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_evalmn_${SLURM_JOB_ID}_${SLURM_NODEID}
      export TMPDIR=/tmp/tmp_'"$USER"'_evalmn_${SLURM_JOB_ID}_${SLURM_NODEID}
      mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
      cd '"$REPO"'
      python infer_helios.py \
        --base_model_path '"$BASE_MODEL"' --transformer_path '"$BASE_MODEL"' '"$lora_str"' \
        --sample_type t2v --prompt_txt_path '"$prompts"' \
        --num_frames '"$NUM_FRAMES"' --fps 24 \
        --num_inference_steps '"$NUM_STEPS"' --guidance_scale '"$GUIDANCE"' \
        --height '"$H"' --width '"$W"' --output_folder '"$out"'
    '
  else
    # Single-node torchrun (EVAL_GPUS skips c-node08 GPU1 which thermally throttles & wedges its rank).
    local gpus=${EVAL_GPUS:-$(seq -s, 0 $((NGPU-1)))}
    local ngpu=$(echo "$gpus" | tr ',' '\n' | grep -c .)
    echo "[gen] using GPUs: $gpus ($ngpu procs)"
    CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=$ngpu --master_port=29611 \
      infer_helios.py \
        --base_model_path "$BASE_MODEL" \
        --transformer_path "$BASE_MODEL" \
        "${lora_args[@]}" \
        --sample_type t2v \
        --prompt_txt_path "$prompts" \
        --num_frames "$NUM_FRAMES" --fps 24 \
        --num_inference_steps "$NUM_STEPS" --guidance_scale "$GUIDANCE" \
        --height "$H" --width "$W" \
        ${ENABLE_COMPILE:+--enable_compile} \
        --output_folder "$out"
  fi
  # rename {idx}.mp4 -> {idx}_{nf}_ori{nf}.mp4 (HeliosBench globs *_*_ori*.mp4), copy meta CSV
  for f in "$out"/*.mp4; do
    b=$(basename "$f" .mp4)
    case "$b" in *_ori*) ;; *_*_*) ;; *) mv "$f" "$out/${b}_${NUM_FRAMES}_ori${NUM_FRAMES}.mp4" ;; esac
  done
  cp "$EVAL_PROMPT_DIR/eval_meta_${set}.csv" "$out/_input.csv"
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
  if [ -n "${NATURALNESS_API_KEY:-}" ]; then
    python 4_get_naturalness.py --input_csv "$csv" --video_dir "$vdir" --output_path "$odir" \
        --api_key "$NATURALNESS_API_KEY" --base_url "${NATURALNESS_BASE_URL:-}" --num_workers 16 &
    python 8_get_drifting_naturalness.py --input_csv "$csv" --video_dir "$vdir" --output_path "$odir" \
        --api_key "$NATURALNESS_API_KEY" --base_url "${NATURALNESS_BASE_URL:-}" --num_workers 16 &
  fi
  wait
  python 9_merge_all_scores.py --input_dir "$odir" --is_long || true
  echo "[metrics] done -> $odir"
}

SETS=()
case "$WHICH" in test) SETS=(test);; train) SETS=(train);; both) SETS=(test train);; esac
for s in "${SETS[@]}"; do gen_one "$s"; done
if [ "$RUN_METRICS" = "1" ]; then for s in "${SETS[@]}"; do run_metrics "$s"; done; fi
echo "[run_eval_once] step=$STEP complete."

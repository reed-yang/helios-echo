#!/bin/bash
# LOCAL (no-Slurm) live-eval watcher for c-node08. Runs as a detached background process directly on
# c-node08's GPUs (torchrun data-parallel). Polls the full-FT and LoRA checkpoint dirs; for each NEW
# checkpoint (skip-to-latest per run) generates 15 sample ~2-min videos from a fixed training-set prompt
# set. full-FT: consolidate the DeepSpeed checkpoint -> checkpoint-N/transformer/ then infer; LoRA: infer
# with the adapter. Shares c-node08 GPUs with whatever else runs there (uses ~50GB/GPU on top).
#
# Launch detached:  setsid nohup bash scripts/eval_live/eval_cfr_local.sh > ~/.eval_cfr.log 2>&1 &
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY=/mnt/beegfs/yuheng/miniconda3/envs/helios
export PATH=$PY/bin:$PATH
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_LEVEL=NVL NCCL_DEBUG=WARN
export WANDB_MODE=disabled   # eval is for sample videos; no wandb run

SNAP=/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
PROMPTS=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/prompts.txt
OUT_ROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr
NUM_FRAMES=${NUM_FRAMES:-2904}   # ~2 min @24fps (88 chunks*33)
STEPS=${STEPS:-50}; GUID=${GUID:-5.0}; H=384; W=640
GPUS=${GPUS:-0,2,3,4,5,6,7}      # skip GPU1 (runs hot); shares node with other jobs
NPROC=$(awk -F',' '{print NF}' <<<"$GPUS")
POLL=${POLL:-300}
mkdir -p "$OUT_ROOT/fullft" "$OUT_ROOT/lora"
echo "[eval_cfr] start $(date) gpus=$GPUS nproc=$NPROC num_frames=$NUM_FRAMES prompts=$PROMPTS"

newest_complete() {  # $1=train_dir  $2=kind(fullft|lora) -> echoes newest checkpoint step whose weights are fully written
  local td=$1 kind=$2 best=-1 n d
  for d in "$td"/checkpoint-*; do
    [ -d "$d" ] || continue
    n=$(basename "$d" | sed 's/checkpoint-//'); [[ "$n" =~ ^[0-9]+$ ]] || continue
    if [ "$kind" = lora ]; then
      [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    else
      [ -f "$d/pytorch_model/mp_rank_00_model_states.pt" ] || continue   # deepspeed save finished
    fi
    [ "$n" -gt "$best" ] && best=$n
  done
  echo "$best"
}

gen() {  # $1=kind  $2=ckpt_dir  $3=step
  local kind=$1 ckpt=$2 step=$3 out="$OUT_ROOT/$1/step-$3" tf lora=""
  mkdir -p "$out"
  if [ "$kind" = fullft ]; then
    # Consolidate DeepSpeed shards -> ckpt/transformer (idempotent; CPU).
    CUDA_VISIBLE_DEVICES="" python "$REPO/tools/offload_data/consolidate_fullft.py" "$ckpt" "$SNAP" || { echo "[eval_cfr] consolidate failed $ckpt"; return 1; }
    tf="$ckpt"
  else
    tf="$EVAL_BASE"; lora="--lora_path $ckpt"
  fi
  echo "[eval_cfr] $(date) GEN $kind step=$step -> $out"
  CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node="$NPROC" --master_port=$((29610 + RANDOM % 50)) \
    "$REPO/infer_helios.py" \
    --base_model_path "$EVAL_BASE" --transformer_path "$tf" $lora \
    --prompt_txt_path "$PROMPTS" \
    --sample_type t2v --num_frames "$NUM_FRAMES" --fps 24 \
    --num_inference_steps "$STEPS" --guidance_scale "$GUID" --height "$H" --width "$W" \
    --output_folder "$out" \
    && echo "[eval_cfr] DONE $kind step=$step videos=$(ls "$out"/*.mp4 2>/dev/null | wc -l)" \
    || echo "[eval_cfr] infer nonzero rc $kind step=$step"
}

declare -A TRAIN=( [fullft]=/mnt/beegfs/xiangbo/helios_runs/stage1_fullft_cfr [lora]=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr )
while true; do
  for kind in fullft lora; do
    td=${TRAIN[$kind]}; state="$OUT_ROOT/$kind/.last"; last=$(cat "$state" 2>/dev/null || echo -1)
    step=$(newest_complete "$td" "$kind")
    if [ "$step" -ge 0 ] && [ "$step" -gt "$last" ]; then
      echo "[eval_cfr] new $kind checkpoint-$step (last=$last)"
      if gen "$kind" "$td/checkpoint-$step" "$step"; then echo "$step" > "$state"; fi
    fi
  done
  sleep "$POLL"
done

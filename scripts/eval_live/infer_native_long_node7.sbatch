#!/bin/bash
#SBATCH --job-name=helios_native_long
#SBATCH --nodes=1
#SBATCH --nodelist=c-node07
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/native_long_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/native_long_%j.out
# ONE-SHOT long-gen (~91s / 2178f) inference of the NATIVE released models only:
#   - Helios-Mid       (Stage2 pyramid, 20/20/20 steps, CFG 5.0)
#   - Helios-Distilled (Stage2 pyramid,  2/2/2  steps, CFG-free 1.0, amplify_first_chunk)
# NO LoRA / NO fine-tune. Same 20 test + 20 train short-caption prompts as the vs24 long eval.
# 8-GPU single-node data-parallel on c-node07 (all 8 H200 idle there; infer_helios shards
# prompts by torchrun rank: prompt_list[rank::world_size]).
# Launch: sbatch scripts/eval_live/infer_native_long_node7.sbatch
#   (c-node07 must be RESUMEd first — it is currently DRAIN'd; see chat.)
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

export REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
export PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PATH=$PY_ENV/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}
# Node-local kernel caches (8 concurrent compile procs race on a beegfs cache).
export TRITON_CACHE_DIR=/tmp/triton_${USER}_natlong_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_natlong_${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

MID=BestWishYsh/Helios-Mid                                   # in xiangbo HF cache
DISTILLED=/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled  # local full snapshot
PROMPT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_vs24_long
OUT_ROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_native_long
NUM_FRAMES=2178 FPS=24 H=384 W=640 NGPU=8

cd "$REPO"

run_model() {  # $1=label  $2=model_path  $3...=extra infer args
  local label=$1 model=$2; shift 2
  for set in test train; do
    local prompts=$PROMPT_DIR/eval_prompts_${set}.txt
    local out=$OUT_ROOT/$label/$set
    mkdir -p "$out"
    echo "=== [$label/$set] $(wc -l < "$prompts") prompts x ${NUM_FRAMES}f -> $out  ($(date)) ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=$NGPU --master_port=29617 \
      infer_helios.py \
        --base_model_path "$model" --transformer_path "$model" \
        --sample_type t2v --prompt_txt_path "$prompts" \
        --num_frames "$NUM_FRAMES" --fps "$FPS" --height "$H" --width "$W" \
        --output_folder "$out" \
        "$@"
  done
}

# 1) Native Helios-Mid: pyramid 20/20/20, CFG 5.0, zero-init first step.
run_model helios-mid "$MID" \
  --is_enable_stage2 --pyramid_num_inference_steps_list 20 20 20 \
  --guidance_scale 5.0 --use_zero_init --zero_steps 1

# 2) Native Helios-Distilled: pyramid 2/2/2, CFG-free (1.0), amplify first chunk.
run_model helios-distilled "$DISTILLED" \
  --is_enable_stage2 --pyramid_num_inference_steps_list 2 2 2 \
  --guidance_scale 1.0 --is_amplify_first_chunk

echo "NATIVE_LONG DONE $(date +%s)"

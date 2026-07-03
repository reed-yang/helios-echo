#!/bin/bash
#SBATCH --job-name=helios_infer_wan22
#SBATCH --nodes=1
#SBATCH --nodelist=c-node07          # INFERENCE node (user-assigned)
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/infer_wan22_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/infer_wan22_%j.out
# t2v inference of a Wan2.2-5B Helios checkpoint (Stage-1 Base: flow matching, 50 steps, CFG 5.0).
# Data-parallel over GPUs (infer_helios shards prompts by torchrun rank). Generates .mp4 into OUT.
#
#   TRANSFORMER=/mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22_merged/transformer \
#   LABEL=init5500 OUT=/mnt/beegfs/xiangbo/helios_runs/eval_out_wan22/init5500 \
#   sbatch scripts/wan22/infer_wan22_node7.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

export REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
export PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PATH=$PY_ENV/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_${USER}_infwan22_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_infwan22_${SLURM_JOB_ID}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Wan2.2 base supplies vae/text_encoder/tokenizer; TRANSFORMER is the merged init/post transformer dir.
BASE=${BASE:-/mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers}
TRANSFORMER=${TRANSFORMER:-/mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22_merged/transformer}
LABEL=${LABEL:-wan22}
OUT=${OUT:-/mnt/beegfs/xiangbo/helios_runs/eval_out_wan22/$LABEL}
PROMPTS=${PROMPTS:-/mnt/beegfs/xiangbo/helios_runs/eval_prompts_vs24_long/eval_prompts_test.txt}
NUM_FRAMES=${NUM_FRAMES:-99}
NGPU=${NGPU:-4}
H=${H:-384}; W=${W:-640}; FPS=${FPS:-24}
DEV=$(seq -s, 0 $((NGPU-1)))
mkdir -p "$OUT"
echo "infer_wan22: base=$BASE tf=$TRANSFORMER out=$OUT prompts=$PROMPTS frames=$NUM_FRAMES ngpu=$NGPU"

cd "$REPO"
CUDA_VISIBLE_DEVICES=$DEV torchrun --nproc_per_node=$NGPU --master_port=29619 \
  infer_helios.py \
    --base_model_path "$BASE" --transformer_path "$TRANSFORMER" \
    --sample_type t2v --prompt_txt_path "$PROMPTS" \
    --num_frames "$NUM_FRAMES" --fps "$FPS" --height "$H" --width "$W" \
    --num_inference_steps 50 --guidance_scale 5.0 --num_latent_frames_per_chunk 9 \
    --output_folder "$OUT"
echo "INFER_WAN22 DONE $(date +%s); videos: $(ls -U "$OUT" 2>/dev/null | grep -c '\.mp4')"

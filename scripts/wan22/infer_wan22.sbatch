#!/bin/bash
#SBATCH --job-name=helios_infer_wan22
#SBATCH --nodes=1
#SBATCH --nodelist=c-node08          # INFERENCE/eval node (node8; GPU1 Prohibited -> auto-skipped)
#SBATCH --gres=gpu:h200:7
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/infer_wan22_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/infer_wan22_%j.out
# t2v inference of a Wan2.2-5B Helios checkpoint (Base: flow matching, 50 steps, CFG 5.0) on node8.
# Data-parallel over the node8 Default GPUs (prohibited GPU1 auto-dropped). Generates .mp4 into OUT.
#   TRANSFORMER=/mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22full_merged/transformer \
#   LABEL=init5500 sbatch scripts/wan22/infer_wan22.sbatch
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

BASE=${BASE:-/mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers}
TRANSFORMER=${TRANSFORMER:-/mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22full_merged}  # ROOT dir; infer_helios appends subfolder="transformer"
LABEL=${LABEL:-wan22full}
OUT=${OUT:-/mnt/beegfs/xiangbo/helios_runs/eval_out_wan22/$LABEL}
# Multi-event prompt-SWITCHING eval: the 5 canonical qb cases (ids 1000/1001/1002/1003/1010),
# 6 events each, combined in evtsw.csv. Each event lasts interpolate_time=7 chunks -> 6*7=42
# chunks = 1386 frames (~1 min). Produces <id>.mp4 per case.
CSV=${CSV:-/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/evtsw.csv}
INTERP_TIME=${INTERP_TIME:-7}
NUM_FRAMES=${NUM_FRAMES:-1386}; H=${H:-384}; W=${W:-640}; FPS=${FPS:-24}
mkdir -p "$OUT"

# Node8: use the Default (non-Prohibited) GPUs; renumber to a contiguous 0..N-1 for torchrun.
GOOD=$(nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader | awk -F", " '$2=="Default"{print $1}' | paste -sd,)
NGPU=$(echo "$GOOD" | tr ',' '\n' | grep -c .)
export CUDA_VISIBLE_DEVICES="$GOOD"
echo "infer_wan22: node8 good GPUs=$GOOD (n=$NGPU) tf=$TRANSFORMER out=$OUT frames=$NUM_FRAMES"

cd "$REPO"
torchrun --nproc_per_node=$NGPU --master_port=29621 \
  infer_helios.py \
    --base_model_path "$BASE" --transformer_path "$TRANSFORMER" \
    --interactive_prompt_csv_path "$CSV" \
    --use_interpolate_prompt --interpolation_steps 0 --interpolate_time "$INTERP_TIME" \
    --num_frames "$NUM_FRAMES" --fps "$FPS" --height "$H" --width "$W" \
    --num_inference_steps 50 --guidance_scale 5.0 --num_latent_frames_per_chunk 9 \
    --output_folder "$OUT"
echo "INFER_WAN22 DONE $(date +%s); videos: $(ls -U "$OUT" 2>/dev/null | grep -c '\.mp4')"

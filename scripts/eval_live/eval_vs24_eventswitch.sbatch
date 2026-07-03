#!/bin/bash
#SBATCH --job-name=helios_evtsw
#SBATCH --nodes=1
#SBATCH --exclude=c-node07,c-node08   # c-node07 drained-ish / c-node08 GPU1 hot; take any other good node
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/evtsw_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/evtsw_%j.out
# EVENT-SWITCHING long inference for the vs24 model: drive infer_helios.py's interactive
# prompt-switching mode (--interactive_prompt_csv_path) over the wander CSV. Each video has 6
# events; each event = 7 chunks (~9.6s @24fps) -> 42 chunks = 1386 frames (~58s, "~1 min").
# Hard switch at event boundaries (interpolation_steps=0). 8-GPU data-parallel (infer_helios
# shards video ids by rank). One-shot over the LATEST vs24 checkpoint. Launch:
#   sbatch scripts/eval_live/eval_vs24_eventswitch.sbatch        # auto-picks latest ckpt
#   CKPT=checkpoint-5500 sbatch scripts/eval_live/eval_vs24_eventswitch.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
TRAIN_OUT=/mnt/beegfs/xiangbo/helios_runs/stage1_video_single_24fps
BASE_MODEL=BestWishYsh/Helios-Base
CSV=$REPO/example/prompt_interactive_helios_wander.csv
# Latest checkpoint unless CKPT overridden.
CKPT=${CKPT:-$(ls -d "$TRAIN_OUT"/checkpoint-* 2>/dev/null | grep -oE 'checkpoint-[0-9]+' | sort -t- -k2 -n | tail -1)}
CKPT_DIR=$TRAIN_OUT/$CKPT
OUT=/mnt/beegfs/xiangbo/helios_runs/eval_out_vs24_eventswitch/$CKPT
mkdir -p "$OUT"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29617
echo "evtsw: ckpt=$CKPT dir=$CKPT_DIR out=$OUT node=$SLURM_JOB_NODELIST csv=$CSV"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false
  export NCCL_DEBUG=WARN
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR='"${MASTER_ADDR}"' MASTER_PORT='"${MASTER_PORT}"'
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_evtsw_${SLURM_JOB_ID}
  export TMPDIR=/tmp/tmp_'"$USER"'_evtsw_${SLURM_JOB_ID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
  cd '"$REPO"'
  python infer_helios.py \
    --base_model_path '"$BASE_MODEL"' --transformer_path '"$BASE_MODEL"' \
    --lora_path '"$CKPT_DIR"' \
    --interactive_prompt_csv_path '"$CSV"' \
    --num_frames 1386 --fps 24 \
    --num_inference_steps 50 --guidance_scale 5.0 \
    --height 384 --width 640 \
    --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
    --output_folder '"$OUT"'
'
echo "EVTSW DONE $(date +%s); videos: $(ls ${OUT}/*.mp4 2>/dev/null | wc -l)"

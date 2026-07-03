#!/bin/bash
#SBATCH --job-name=helios_enc_wan22
#SBATCH --nodes=1
#SBATCH --nodelist=c-node08          # node8 default; now 8 usable GPUs (GPU1 prohibited-state cleared). dynamic-CVD drops any that are Prohibited
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0                       # all node RAM: bprime clips run up to 501 frames -> big CPU batches
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/enc_wan22_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/enc_wan22_%j.out
# Encode cfr_int clips -> 384x640 latents with the Wan2.2-TI2V-5B VAE (z_dim=48, 16x spatial).
# Output .pt: vae_latent (num_chunks,48,9,24,40) + prompt_embed_short + legacy prompt_embed.
# Idempotent (skips existing .pt) -> resubmit to continue. One srun task per GPU + env rendezvous.
#
#   # smoke (~200):
#   sbatch JSON=$DATA/offload_cfr_int_wan22_smoke.json OUT=$DATA/latents_wan22_384x640_smoke \
#          scripts/wan22/encode_wan22.sbatch
#   # PoC (~25k):
#   sbatch JSON=$DATA/offload_cfr_int_wan22_poc.json OUT=$DATA/latents_wan22_384x640_poc \
#          scripts/wan22/encode_wan22.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
DATA=/mnt/beegfs/dataset/video_single_24FPS
JSON=${JSON:-$DATA/offload_cfr_int_wan22_poc.json}
VIDEO_DIR=${VIDEO_DIR:-$DATA/videos_cfr_int}
OUT=${OUT:-$DATA/latents_wan22_384x640_poc}
BATCH_SIZE=${BATCH_SIZE:-2}
# Wan2.2-TI2V-5B-Diffusers provides vae/ (z_dim=48), text_encoder/ (umT5-xxl), tokenizer/.
ENCODER_SRC=${ENCODER_SRC:-/mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers}
mkdir -p "$OUT"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29547
export WORLD_SIZE=$SLURM_NTASKS
echo "enc_wan22: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT out=$OUT json=$JSON encoder=$ENCODER_SRC"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false
  # These cfr_int/bprime videos are known to exist (they produced the 368 latents) -> skip the
  # slow beegfs os.walk existence scan in the dataset build (huge speedup on the 280k bprime set).
  export HELIOS_ASSUME_VIDEOS_EXIST=1
  export HELIOS_READ_TIMEOUT=90         # skip corrupt videos that hang the reader (else a bad clip freezes the job)
  export HELIOS_ENC_FREE_EVERY=4        # throttle gc/empty_cache; 4 keeps GPU-util win but stays memory-bounded; 25 caused OOM
  # Node-agnostic GPU selection: keep the Slurm-allocated GPUs, dropping any in Prohibited compute
  # mode (node8 keeps GPU1 hot for inference). Works on node7 (8 Default) and node8 (6 Default).
  PROHIB=$(nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader | awk -F", " "\$2!=\"Default\"{print \$1}")
  NEW=""; for g in ${CUDA_VISIBLE_DEVICES//,/ }; do keep=1; for p in $PROHIB; do [ "$g" = "$p" ] && keep=0; done; [ $keep -eq 1 ] && NEW="$NEW,$g"; done
  export CUDA_VISIBLE_DEVICES="${NEW#,}"
  echo "rank $SLURM_PROCID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (dropped prohibited: ${PROHIB:-none})"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_encwan22_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_encwan22_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_encwan22_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python tools/offload_data/get_short-latents.py \
    --pretrained_model_name_or_path '"$ENCODER_SRC"' \
    --json_file '"$JSON"' \
    --video_folder '"$VIDEO_DIR"' \
    --output_latent_folder '"$OUT"' \
    --caption_versions short \
    --single_height 384 --single_width 640 \
    --resolution 642 \
    --batch_size '"$BATCH_SIZE"'
'
echo "ENC_WAN22 DONE $(date +%s); pt files: $(ls -U ${OUT} 2>/dev/null | grep -c "\.pt")"

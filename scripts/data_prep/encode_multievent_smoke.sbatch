#!/bin/bash
#SBATCH --job-name=helios_enc_me_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c-node08
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/encode_me_smoke_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/encode_me_smoke_%j.out
# 8-GPU smoke encode of a small multi-event subset (multievent_smoke_train.json) -> latents_multievent_smoke/.
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
DATA=/mnt/beegfs/dataset/streaming_model_data
JSON_FILE=${JSON_FILE:-$DATA/multievent_smoke_train.json}
OUT=${OUT:-$DATA/latents_multievent_smoke}
ENCODER_SRC=/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled
mkdir -p "$OUT"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29533
export WORLD_SIZE=$SLURM_NTASKS
echo "encode_me_smoke: ntasks=$SLURM_NTASKS json=$JSON_FILE out=$OUT"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_encmesmoke_${SLURM_JOB_ID}_${SLURM_NODEID}
  cd '"$REPO"'
  python tools/offload_data/get_multievent-latents.py \
    --pretrained_model_name_or_path '"$ENCODER_SRC"' \
    --json_file '"$JSON_FILE"' \
    --video_folder '"$DATA"'/videos_24fps \
    --output_latent_folder '"$OUT"' \
    --batch_size 2
'
echo "ENCODE_ME_SMOKE DONE $(date +%s); pt files: $(ls ${OUT}/*.pt 2>/dev/null | wc -l)"

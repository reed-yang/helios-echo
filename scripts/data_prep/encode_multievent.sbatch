#!/bin/bash
#SBATCH --job-name=helios_enc_me
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8          # one task per GPU -> 32 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c-node08
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/encode_me_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/encode_me_%j.out
# Encode the ~13.2k multi-event train clips -> 384x640 latents with per-event prompt embeds
# + chunk->event map (.pt). One srun task per GPU + env-based rendezvous (nested
# torch.distributed.run fails to bind its c10d store across nodes here). Idempotent.
# Override JSON_FILE / OUT via env to run the smoke subset on the same launcher.
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
DATA=/mnt/beegfs/dataset/streaming_model_data
JSON_FILE=${JSON_FILE:-$DATA/multievent_ohv_train.json}
OUT=${OUT:-$DATA/latents_multievent_384x640}
# Encoder components from a PLAIN LOCAL DIR (not an HF repo id): 32 concurrent ranks racing
# on HF Hub snapshot resolution intermittently report missing shards. Same Wan2.1 vae/text_encoder.
ENCODER_SRC=/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled
mkdir -p "$OUT"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29532
export WORLD_SIZE=$SLURM_NTASKS
echo "encode_me: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT json=$JSON_FILE out=$OUT"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_encme_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_encme_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
  cd '"$REPO"'
  python tools/offload_data/get_multievent-latents.py \
    --pretrained_model_name_or_path '"$ENCODER_SRC"' \
    --json_file '"$JSON_FILE"' \
    --video_folder '"$DATA"'/videos_24fps \
    --output_latent_folder '"$OUT"' \
    --batch_size 2
'
echo "ENCODE_ME DONE $(date +%s); pt files: $(ls ${OUT}/*.pt 2>/dev/null | wc -l)"

#!/bin/bash
#SBATCH --job-name=helios_encode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8          # one task per GPU -> 32 tasks; srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --exclude=c-node08
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/encode_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/encode_%j.out
# Encode all 14,651 clips -> 384x640 latents (.pt w/ 4 caption embeds) on 32 GPUs.
# Uses one srun task per GPU + env-based rendezvous (NO nested torch.distributed.run,
# which failed to bind its c10d store across nodes). Idempotent (skips existing .pt).
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
DATA=/mnt/beegfs/dataset/streaming_model_data
OUT=$DATA/latents_short_384x640
# Encoder components (tokenizer/vae/text_encoder) loaded from a PLAIN LOCAL DIR, not an HF
# cache repo id: 32 concurrent ranks racing on the HF Hub snapshot resolution intermittently
# report missing shards (OSError ... model-0000N-of-00005.safetensors). A flat local dir has
# no Hub resolution/locking. This Distilled copy has the same Wan2.1 vae/text_encoder/tokenizer.
ENCODER_SRC=/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled
mkdir -p "$OUT"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29531
export WORLD_SIZE=$SLURM_NTASKS
echo "encode: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT nodelist=$SLURM_JOB_NODELIST out=$OUT"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  # env that the script'\''s dist.init_process_group + Accelerator read:
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_enc_${SLURM_JOB_ID}_${SLURM_NODEID}
  cd '"$REPO"'
  python tools/offload_data/get_short-latents.py \
    --pretrained_model_name_or_path '"$ENCODER_SRC"' \
    --json_file '"$DATA"'/offload_train_24fps.json \
    --video_folder '"$DATA"'/videos_24fps \
    --output_latent_folder '"$OUT"' \
    --rewritten_json '"$DATA"'/prompts_rewritten.json \
    --batch_size 2
'
echo "ENCODE DONE $(date +%s); pt files: $(ls ${OUT}/*.pt 2>/dev/null | wc -l)"

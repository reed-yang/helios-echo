#!/bin/bash
#SBATCH --job-name=helios_merge368
#SBATCH --nodes=1
#SBATCH --nodelist=c-node05
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:50:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/merge368_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/merge368_%j.out
# Selective merge: LoRA-17000 (368x640) -> consolidated start for the full-base + LoRA-patch run.
# CPU-only (loads 14B fp32 ~57GB, merges 407 Linear LoRAs, keeps patch_embedding LoRA separate).
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
export PATH=$PY_ENV/bin:$PATH
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
cd "$REPO"
CKPT=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368/checkpoint-17000
python tools/merge_lora_partial_for_helios.py \
  --base /mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd \
  --base_subfolder transformer_init \
  --lora  "$CKPT/pytorch_lora_weights.safetensors" \
  --partial "$CKPT/transformer_partial.pth" \
  --out /mnt/beegfs/xiangbo/helios_runs/_merged/stage1_lora368_merged17000 \
  --lora_alpha 128 --lora_rank 128
echo "MERGE368 DONE $(date +%s)"

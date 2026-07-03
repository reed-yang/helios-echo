#!/bin/bash
#SBATCH --job-name=helios_merge_wan22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/merge_wan22_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/merge_wan22_%j.out
# Merge a trained Stage-1 LoRA (+ transformer_partial.pth) into its transformer base -> standalone
# transformer/. CPU-only (avoids node8's Prohibited GPU). Auto-detects the *-final (or highest) checkpoint.
#
#   # init merge (base = raw Wan2.2):
#   CKPT_ROOT=/mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22 \
#   OUT=/mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22_merged sbatch scripts/wan22/merge_wan22.sbatch
#   # post merge (base = merged-init, vae/text from Wan2.2):
#   BASE=/mnt/beegfs/xiangbo/helios_runs/stage1_init_wan22_merged \
#   CKPT_ROOT=/mnt/beegfs/xiangbo/helios_runs/stage1_post_wan22 \
#   OUT=/mnt/beegfs/xiangbo/helios_runs/stage1_post_wan22_merged sbatch scripts/wan22/merge_wan22.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
WAN22=/mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers

BASE=${BASE:-$WAN22}                 # transformer base the LoRA was trained on
PIPE_BASE=${PIPE_BASE:-$WAN22}       # vae/text/tokenizer source (always Wan2.2)
CKPT_ROOT=${CKPT_ROOT:?set CKPT_ROOT to the training output dir}
OUT=${OUT:?set OUT}

# Resolve checkpoint: prefer *-final, else highest checkpoint-N
CKPT=$(ls -d "$CKPT_ROOT"/checkpoint-*-final 2>/dev/null | head -1 || true)
[ -z "$CKPT" ] && CKPT=$(ls -d "$CKPT_ROOT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
[ -z "$CKPT" ] && { echo "no checkpoint found under $CKPT_ROOT"; exit 1; }
echo "merge_wan22: base=$BASE pipe_base=$PIPE_BASE ckpt=$CKPT out=$OUT"

export PATH=$PY_ENV/bin:$PATH
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""        # CPU-only merge
cd "$REPO"
python scripts/wan22/merge_lora_wan22.py \
  --base "$BASE" --pipe_base "$PIPE_BASE" \
  --lora "$CKPT/pytorch_lora_weights.safetensors" \
  --partial "$CKPT/transformer_partial.pth" \
  --out "$OUT"
echo "MERGE_WAN22 DONE $(date +%s); merged: $(ls "$OUT"/transformer 2>/dev/null | head)"

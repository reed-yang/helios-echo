#!/bin/bash
#SBATCH --job-name=enc_chunkcap
#SBATCH --nodes=6
#SBATCH --nodelist=mc-node01,c-node03,c-node04,c-node05,c-node06,c-node07
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/enc_chunkcap_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/enc_chunkcap_%j.out
# Deterministic chunk-aligned re-encode (docs/CHUNK_ALIGNED_CAPTIONING.md pitfall #11):
# frames [cut_start, cut_start + n_chunks*33) at 368x640 + per-chunk UMT5 embeds.
# Nodes = node1,3,4,5,6,7 per user: mc-node02 and c-node08 have hot GPUs that throttle and
# drag multi-GPU jobs (c-node08 also hosts an out-of-Slurm vLLM). Idempotent (skips existing .pt).
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
DATA=/mnt/beegfs/dataset/video_single_24FPS
CAPS=$DATA/chunk_captions_gemini_30k/captions_chunks.jsonl
HELD=$DATA/chunk_captions_gemini_30k/heldout_100.txt
OUT=${OUT:-$DATA/latents_chunkcap_368x640}
ENCODER_SRC=/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled
mkdir -p "$OUT"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29533
export WORLD_SIZE=$SLURM_NTASKS
echo "enc_chunkcap: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT out=$OUT"

srun --mpi=none --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_encc_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_encc_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
  cd '"$REPO"'
  python scripts/data_prep/chunk_captioning/encode_chunk_latents.py \
    --captions '"$CAPS"' \
    --exclude '"$HELD"' \
    --outdir '"$OUT"' \
    --pretrained_model_name_or_path '"$ENCODER_SRC"'
'
echo "ENC_CHUNKCAP DONE $(date +%s); pt files: $(ls ${OUT}/*.pt 2>/dev/null | wc -l)"

#!/bin/bash
#SBATCH --job-name=helios_enccfr
#SBATCH --ntasks-per-node=8          # one task per GPU -> srun sets RANK/LOCAL_RANK
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/enccfr_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/enccfr_%j.out
# Encode the cfr_int / captions_30b_32f train set -> 384x640 latents (.pt with prompt_embed_short +
# legacy prompt_embed, both = encode(single manifest prompt)). Idempotent: skips clips whose .pt
# exists, so a timed-out/killed job can simply be resubmitted to continue. One srun task per GPU +
# env rendezvous. Source videos = videos_cfr_int (read VERBATIM via the offload JSON 'path' basename).
#
# Submit (pick --nodes/--nodelist on the CLI):
#   sbatch --nodes=1 --nodelist=c-node05 JSON=$DATA/offload_cfr_int_smoke200.json \
#          OUT=$DATA/latents_cfr_int_smoke scripts/data_prep/encode_cfr_int.sbatch     # smoke
#   sbatch --nodes=4 --nodelist=mc-node01,mc-node02,c-node03,c-node05 \
#          scripts/data_prep/encode_cfr_int.sbatch                                     # full
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
DATA=/mnt/beegfs/dataset/video_single_24FPS
JSON=${JSON:-$DATA/offload_cfr_int.json}
VIDEO_DIR=${VIDEO_DIR:-$DATA/videos_cfr_int}
OUT=${OUT:-$DATA/latents_cfr_int_30b_384x640}
BATCH_SIZE=${BATCH_SIZE:-2}
# Encoder components (tokenizer/vae/text_encoder) from a PLAIN LOCAL DIR (not an HF repo id):
# 32 concurrent ranks racing on HF-Hub snapshot resolution intermittently report missing shards.
ENCODER_SRC=/mnt/beegfs/yuheng/Helios/BestWishYSH/Helios-Distilled
mkdir -p "$OUT"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29543
export WORLD_SIZE=$SLURM_NTASKS
echo "enccfr: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT nodelist=$SLURM_JOB_NODELIST out=$OUT json=$JSON"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_enccfr_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_enccfr_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_enccfr_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  # Serialize gcc per node (beegfs concurrent conda-header reads transiently ENOENT -> gcc exit 1).
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python tools/offload_data/get_short-latents.py \
    --pretrained_model_name_or_path '"$ENCODER_SRC"' \
    --json_file '"$JSON"' \
    --video_folder '"$VIDEO_DIR"' \
    --output_latent_folder '"$OUT"' \
    --caption_versions short \
    --batch_size '"$BATCH_SIZE"'
'
echo "ENCCFR DONE $(date +%s); pt files: $(ls -U ${OUT} 2>/dev/null | grep -c "\.pt")"

#!/bin/bash
#SBATCH --job-name=helios_tagemb
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=c-node07
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/tagemb_%A_%a.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/tagemb_%A_%a.out
# Encode tagged-caption UMT5 sidecars for the reweight_v2 (rwtag) run.
# 8 single-GPU array workers on c-node07 (idle; multi-node NCCL not involved),
# each takes manifest rows idx % 8 == ARRAY_ID. Idempotent (skips existing).
# Launch: sbatch scripts/data/encode_tag_sidecars.sbatch
set -euo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
export PATH=$PY_ENV/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH=$REPO:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled HF_ENABLE_PARALLEL_LOADING=no
cd "$REPO"
python scripts/data/encode_tag_sidecars.py --shard "$SLURM_ARRAY_TASK_ID" --num_shards 8

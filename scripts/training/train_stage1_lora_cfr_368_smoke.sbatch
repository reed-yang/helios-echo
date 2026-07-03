#!/bin/bash
#SBATCH --job-name=helios_lora368sm
#SBATCH --nodes=1
#SBATCH --nodelist=c-node07            # 1 node = intra-node NCCL only -> c07's multi-node deadlock N/A
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_lora368sm_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_lora368sm_%j.out
# SMOKE: validate resume-from-checkpoint-9000 + 368 data on 1 node (8 GPU) before the full run.
# Uses stage1_lora_cfr_368_smoke.yaml: ~200 symlinked 368 .pt, scratch output_dir, EXPLICIT resume from
# .../stage1_lora_cfr/checkpoint-9000, max_train_steps 9006. Success = load_state restores LoRA+optim+sched
# (only random_states_0.pkl present -> world-size-change tolerated), global_step starts 9000, 368 latents
# flow (latent h46 -> pad48), finite loss, a checkpoint saves.
# Launch:  sbatch scripts/training/train_stage1_lora_cfr_368_smoke.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_lora_cfr_368_smoke.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29511
export WORLD_SIZE=$SLURM_NTASKS
echo "train_lora368sm: ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG nodelist=$SLURM_JOB_NODELIST"

srun --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=offline
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_lora368sm_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_lora368sm_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_lora368sm_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  python train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_LORA368SM DONE $(date +%s)"

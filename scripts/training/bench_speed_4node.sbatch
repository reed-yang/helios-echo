#!/bin/bash
#SBATCH --job-name=helios_bench4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/bench4_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/bench4_%j.out
# 4-node (32-GPU) speed bench: env-DDP, ~60 fresh steps, no resume/ckpt/val. Exposes inter-node NCCL
# all-reduce + aggregate beegfs contention that 1-node can't show. Per-node rank0 samples nvidia-smi.
# Pass CONFIG (abs path), nodelist via --nodelist, and HELIOS_THROTTLE_FREE / HELIOS_DDP_FIND_UNUSED env.
set -uo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:?set CONFIG}
SMI=/mnt/beegfs/xiangbo/helios_runs/_bench368/smi4_${SLURM_JOB_ID}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(( 29700 + (SLURM_JOB_ID % 200) ))
export WORLD_SIZE=$SLURM_NTASKS
echo "bench4: job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST config=$CONFIG master=$MASTER_ADDR:$MASTER_PORT free=${HELIOS_THROTTLE_FREE:-0} find=${HELIOS_DDP_FIND_UNUSED:-1}"
srun --mpi=none --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false HF_ENABLE_PARALLEL_LOADING=no
  export HELIOS_THROTTLE_FREE=${HELIOS_THROTTLE_FREE:-0}
  export HELIOS_DDP_FIND_UNUSED=${HELIOS_DDP_FIND_UNUSED:-1}
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export NCCL_P2P_LEVEL=NVL
  export RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_b4_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_b4_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_b4_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  if [ "$SLURM_LOCALID" = "0" ]; then
    ( nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 \
        > '"$SMI"'_$(hostname).csv 2>/dev/null ) & SMIPID=$!
    python train_helios.py --config '"$CONFIG"'; RC=$?; kill $SMIPID 2>/dev/null; exit $RC
  else
    python train_helios.py --config '"$CONFIG"'
  fi
'
echo "BENCH4 DONE job=$SLURM_JOB_ID $(date +%s)"

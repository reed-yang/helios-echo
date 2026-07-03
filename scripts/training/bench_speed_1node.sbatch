#!/bin/bash
#SBATCH --job-name=helios_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=10
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/bench_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/bench_%j.out
# Speed benchmark: 1 node x 8 GPU, ~90 fresh steps over the 6000-clip bench subset, no resume/ckpt/val.
# Samples nvidia-smi util+mem on local-rank 0 to classify compute- vs IO-bound and capture peak memory.
# Pass CONFIG (abs path to a bench variant yaml) and the node via sbatch --nodelist=<node>.
# Launch:  CONFIG=/.../_bench368/configs/v2_gcoff_bs4.yaml sbatch --nodelist=c-node07 scripts/training/bench_speed_1node.sbatch
set -uo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:?set CONFIG=abs path to a bench variant yaml}
SMI=/mnt/beegfs/xiangbo/helios_runs/_bench368/smi_${SLURM_JOB_ID}.csv

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(( 29600 + (SLURM_JOB_ID % 300) ))
export WORLD_SIZE=$SLURM_NTASKS
echo "bench: job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST config=$CONFIG master=$MASTER_ADDR:$MASTER_PORT"

srun --mpi=none --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=offline
  export TOKENIZERS_PARALLELISM=false
  export HELIOS_THROTTLE_FREE=${HELIOS_THROTTLE_FREE:-0}
  export HELIOS_DDP_FIND_UNUSED=${HELIOS_DDP_FIND_UNUSED:-1}
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  export HF_ENABLE_PARALLEL_LOADING=no
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_bench_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_bench_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_bench_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  cd '"$REPO"'
  if [ "$SLURM_LOCALID" = "0" ]; then
    ( nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 \
        > '"$SMI"' 2>/dev/null ) &
    SMIPID=$!
    python train_helios.py --config '"$CONFIG"'
    RC=$?
    kill $SMIPID 2>/dev/null
    exit $RC
  else
    python train_helios.py --config '"$CONFIG"'
  fi
'
echo "BENCH DONE job=$SLURM_JOB_ID $(date +%s)"

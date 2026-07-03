#!/bin/bash
#SBATCH --job-name=helios_fullbase368
#SBATCH --nodes=2
#SBATCH --nodelist=c-node05,c-node06        # fast adjacent pair (c-node05 was never faulty; the crash was a stale config.json)
#SBATCH --ntasks-per-node=1                  # ONE accelerate launch per node; it spawns 8 local procs
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/train_fullbase368_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/train_fullbase368_%j.out
# HYBRID full fine-tune: full base-linear + LoRA r128 patch_embedding + frozen norm, starting from the
# selectively-merged LoRA-17000 (368x640). DeepSpeed ZeRO-2 (env-launcher DDP won't fit ~14B full base +
# Adam). One srun task per node spawns 8 local ranks; machine_rank=$SLURM_NODEID.
# GLOBAL BATCH 128 = 16 GPU x bs1 x accum8. Idempotent restart: resume_from_checkpoint=latest
# (skip_dataloader_dcp=true -> resume is world-size agnostic; DeepSpeed model+optim world-size locked to 16).
#
# PREREQ: run tools/merge_lora_partial_for_helios.py first to build the merged start dir referenced by the config.
#
#   sbatch scripts/training/train_stage1_fullbase_cfr368.sbatch                         # 16-GPU run
#   sbatch --nodes=1 --nodelist=c-node05 \
#          CONFIG=scripts/training/configs/stage1_fullbase_cfr368_smoke.yaml \
#          scripts/training/train_stage1_fullbase_cfr368.sbatch                         # 8-GPU smoke
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
CONFIG=${CONFIG:-scripts/training/configs/stage1_fullbase_cfr368.yaml}
# DeepSpeed ZeRO-2 WITH a deepspeed-managed WarmupLR (warmup_max_lr hardcoded to 1e-6 in the json;
# keep in sync with the config learning_rate). The scheduler must live in the deepspeed config so its
# state is saved/restored inside the engine checkpoint.
ACC_CFG=${ACC_CFG:-scripts/accelerate_configs/multi_node_zero2_sched_1e6.yaml}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29522
NNODES=$SLURM_NNODES
NPROC=$(( NNODES * 8 ))
echo "train_fullbase368: nodes=$NNODES nproc=$NPROC master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG acc=$ACC_CFG nodelist=$SLURM_JOB_NODELIST"

srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  export PATH='"$PY_ENV"'/bin:$PATH
  export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
  export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export WANDB_MODE=online
  export WANDB_API_KEY=$(cat /mnt/beegfs/xiangbo/.wandb_key 2>/dev/null)
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
  # Skip the per-step free_memory() (gc.collect+empty_cache) that thrashes ~54GB cache back to the driver
  # every step and stalls the GPU. The 3 hot-loop sites in train_helios.py are gated on this. ~-18-26% s/it.
  export HELIOS_THROTTLE_FREE=1
  # node-local Triton/inductor cache + gcc serialization (defeats the beegfs concurrent-header race).
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_fb368_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_'"$USER"'_fb368_${SLURM_JOB_ID}_${SLURM_NODEID}
  export TMPDIR=/tmp/tmp_'"$USER"'_fb368_${SLURM_JOB_ID}_${SLURM_NODEID}
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  export CC='"$REPO"'/scripts/training/gcc_serialized.sh
  export HELIOS_GCC_LOCK=/tmp/helios_gcc_'"$USER"'_${SLURM_JOB_ID}.lock
  export HF_ENABLE_PARALLEL_LOADING=no
  cd '"$REPO"'
  # Stage the 27GB merged transformer to node-local /dev/shm (RAM) ONCE per node, then load from there.
  # 16 ranks x 8 workers = 128 concurrent beegfs readers of the same shards SIGBUS under I/O contention;
  # tmpfs cannot SIGBUS. Idempotent: skipped if already staged (survives crash-resume on the same node).
  SRC=/mnt/beegfs/xiangbo/helios_runs/_merged/stage1_lora368_merged17000
  STAGE=/dev/shm/helios_merged/stage1_lora368_merged17000
  if [ ! -f "$STAGE/transformer/diffusion_pytorch_model.safetensors.index.json" ]; then
    echo "[$(hostname)] staging merged transformer -> $STAGE"
    mkdir -p "$STAGE"
    cp -r "$SRC/transformer" "$STAGE/" && cp "$SRC/patch_embedding_lora.safetensors" "$STAGE/"
    echo "[$(hostname)] staging done"
  else
    echo "[$(hostname)] merged transformer already staged at $STAGE"
  fi
  accelerate launch \
    --config_file '"$ACC_CFG"' \
    --num_machines '"$NNODES"' \
    --num_processes '"$NPROC"' \
    --machine_rank $SLURM_NODEID \
    --main_process_ip '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    train_helios.py --config '"$CONFIG"'
'
echo "TRAIN_FULLBASE368 DONE $(date +%s)"

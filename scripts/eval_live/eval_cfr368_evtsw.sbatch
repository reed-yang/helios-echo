#!/bin/bash
#SBATCH --job-name=helios_eval368
#SBATCH --nodes=2
#SBATCH --nodelist=c-node07,c-node08          # the 2 idle nodes (training is on mc01/02,c03-06)
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/eval368_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/eval368_%j.out
# LIVE event-switch eval for the stage1_lora_cfr_368 run. Evals EVERY saved 368 checkpoint (backfill +
# ongoing), ONE CHECKPOINT PER NODE so c07 & c08 run different checkpoints concurrently; within a node the
# 5 event-switch videos are data-parallel across its 8 GPUs (infer_helios shards csv ids by rank).
# Group B only: 5 one-minute event-switch cases (evtsw.csv, ids 1000/1001/1002/1003/1010), 6 events x
# interpolate_time 7 = 1386 frames, hard prompt-switch. Resolution 368x640 (matches training). Reuses
# infer_helios as-is, base = eval_base_init + lora + transformer_partial.pth. Idempotent (.done + skip mp4).
# Launch:  sbatch scripts/eval_live/eval_cfr368_evtsw.sbatch
set -uo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
CSV_B=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/evtsw.csv
TRAIN_DIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr_368
OUT_B=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368/evtsw
STATE=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr368/.done
mkdir -p "$OUT_B" "$STATE"
NF_B=${NF_B:-1386}; STEPS=${STEPS:-50}; GUID=${GUID:-5.0}; H=${H:-368}; W=${W:-640}
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "[eval368] start $(date) nodes=${NODES[*]} res=${H}x${W} ; one checkpoint per node, all 368 checkpoints"

gen_node() {  # $1=node  $2=step
  local node=$1 step=$2 ckpt="$TRAIN_DIR/checkpoint-$2" b="$OUT_B/step-$2"; mkdir -p "$b"
  srun --nodes=1 --nodelist="$node" --ntasks=8 --ntasks-per-node=8 --gres=gpu:h200:8 --exact --kill-on-bad-exit=0 bash -c '
    export PATH='"$PY_ENV"'/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH='"$REPO"':${PYTHONPATH:-}
    export TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL WANDB_MODE=disabled
    export RANK=$SLURM_LOCALID WORLD_SIZE=8 LOCAL_RANK=$SLURM_LOCALID
    export MASTER_ADDR='"$node"' MASTER_PORT=29658
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_eval368_'"$step"' TMPDIR=/tmp/tmp_'"$USER"'_eval368_'"$step"'
    mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"; cd '"$REPO"'
    python infer_helios.py --base_model_path '"$EVAL_BASE"' --transformer_path '"$EVAL_BASE"' \
      --lora_path '"$ckpt"' --partial_path '"$ckpt"'/transformer_partial.pth \
      --interactive_prompt_csv_path '"$CSV_B"' --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
      --num_frames '"$NF_B"' --num_inference_steps '"$STEPS"' --guidance_scale '"$GUID"' \
      --height '"$H"' --width '"$W"' --fps 24 --output_folder '"$b"'
  '
  echo "[eval368] $(date) DONE step=$step on $node : B=$(ls "$b"/*.mp4 2>/dev/null|wc -l)"
  touch "$STATE/$step"
}

while true; do
  todo=()
  for d in "$TRAIN_DIR"/checkpoint-*; do
    [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    [ -f "$d/transformer_partial.pth" ] || continue
    n=$(basename "$d" | sed 's/checkpoint-//'); [[ "$n" =~ ^[0-9]+$ ]] || continue
    [ -f "$STATE/$n" ] || todo+=("$n")
  done
  if [ ${#todo[@]} -eq 0 ]; then sleep 180; continue; fi
  IFS=$'\n' todo=($(sort -n <<<"${todo[*]}")); unset IFS
  i=0
  while [ $i -lt ${#todo[@]} ]; do
    pids=()
    for ((j=0; j<${#NODES[@]} && i<${#todo[@]}; j++, i++)); do
      echo "[eval368] assign checkpoint-${todo[$i]} -> ${NODES[$j]}"
      gen_node "${NODES[$j]}" "${todo[$i]}" &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
  done
done

#!/bin/bash
#SBATCH --job-name=helios_evalcfr
#SBATCH --nodes=4
#SBATCH --nodelist=mc-node01,mc-node02,c-node03,c-node04   # freed full-FT nodes
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/evalcfr_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/evalcfr_%j.out
# LoRA-only live eval that evaluates EVERY saved checkpoint (backfill + ongoing), one CHECKPOINT PER
# NODE so all 4 nodes run different checkpoints concurrently. Per checkpoint/node: Group A = 5
# training-set prompts -> 2-min videos (no switch); Group B = 5 event-switch videos -> 1-min (rewritten
# wander prompts, hard switch). Each group is data-parallel across that node's 8 GPUs (infer_helios
# shards prompts/ids by rank). Reuses infer_helios as-is. Idempotent (skips existing mp4 + .done marker).
# Launch:  sbatch scripts/eval_live/eval_cfr_watch.sbatch
set -uo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
PROMPTS_A=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/prompts5.txt
CSV_B=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/evtsw.csv
TRAIN_DIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr
OUT_A=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr/lora
OUT_B=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr/lora_evtsw
STATE=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr/.done
mkdir -p "$OUT_A" "$OUT_B" "$STATE"
NF_A=${NF_A:-2904}; NF_B=${NF_B:-1386}; STEPS=${STEPS:-50}; GUID=${GUID:-5.0}
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "[evalcfr] start $(date) nodes=${NODES[*]} ; one checkpoint per node, all checkpoints"

# Generate both groups for ONE checkpoint on ONE node (8 GPU data-parallel). Backgroundable.
gen_node() {  # $1=node  $2=step
  local node=$1 step=$2 ckpt="$TRAIN_DIR/checkpoint-$2"
  local a="$OUT_A/step-$2" b="$OUT_B/step-$2"; mkdir -p "$a" "$b"
  for grp in A B; do
    local args
    if [ "$grp" = A ]; then
      args="--prompt_txt_path $PROMPTS_A --sample_type t2v --num_frames $NF_A --output_folder $a"
    else
      args="--interactive_prompt_csv_path $CSV_B --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 --num_frames $NF_B --output_folder $b"
    fi
    srun --nodes=1 --nodelist="$node" --ntasks=8 --ntasks-per-node=8 --gres=gpu:h200:8 --exact --kill-on-bad-exit=0 bash -c '
      export PATH='"$PY_ENV"'/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH='"$REPO"':${PYTHONPATH:-}
      export TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL WANDB_MODE=disabled
      export RANK=$SLURM_LOCALID WORLD_SIZE=8 LOCAL_RANK=$SLURM_LOCALID
      export MASTER_ADDR='"$node"' MASTER_PORT=29656
      export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
      export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_eval_'"$step"'_'"$grp"' TMPDIR=/tmp/tmp_'"$USER"'_eval_'"$step"'_'"$grp"'
      mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"; cd '"$REPO"'
      python infer_helios.py --base_model_path '"$EVAL_BASE"' --transformer_path '"$EVAL_BASE"' --lora_path '"$ckpt"' \
        --partial_path '"$ckpt"'/transformer_partial.pth \
        --fps 24 --num_inference_steps '"$STEPS"' --guidance_scale '"$GUID"' --height 384 --width 640 '"$args"'
    '
  done
  echo "[evalcfr] $(date) DONE step=$step on $node : A=$(ls "$a"/*.mp4 2>/dev/null|wc -l) B=$(ls "$b"/*.mp4 2>/dev/null|wc -l)"
  touch "$STATE/$step"
}

while true; do
  # collect complete checkpoints not yet done, ascending
  todo=()
  for d in "$TRAIN_DIR"/checkpoint-*; do
    [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
    n=$(basename "$d" | sed 's/checkpoint-//'); [[ "$n" =~ ^[0-9]+$ ]] || continue
    [ -f "$STATE/$n" ] || todo+=("$n")
  done
  if [ ${#todo[@]} -eq 0 ]; then sleep 180; continue; fi
  IFS=$'\n' todo=($(sort -n <<<"${todo[*]}")); unset IFS
  # process in waves of <=#nodes, one checkpoint per node, concurrently
  i=0
  while [ $i -lt ${#todo[@]} ]; do
    pids=()
    for ((j=0; j<${#NODES[@]} && i<${#todo[@]}; j++, i++)); do
      echo "[evalcfr] assign checkpoint-${todo[$i]} -> ${NODES[$j]}"
      gen_node "${NODES[$j]}" "${todo[$i]}" &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
  done
done

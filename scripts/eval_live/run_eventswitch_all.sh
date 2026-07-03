#!/bin/bash
# Sweep ALL vs24 checkpoints and run event-switching long inference on each (idempotent),
# then keep watching for newly-saved checkpoints from the running training. Each checkpoint:
# 41 videos x 6 events (CSV) x 7 chunks (~9.6s) = 1386f (~58s), hard switch. Data-parallel
# across the whole allocation (infer_helios shards video ids by global rank).
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
TRAIN_OUT=/mnt/beegfs/xiangbo/helios_runs/stage1_video_single_24fps
BASE_MODEL=BestWishYsh/Helios-Base
CSV=$REPO/example/prompt_interactive_helios_wander.csv
OUT_ROOT=/mnt/beegfs/xiangbo/helios_runs/eval_out_vs24_eventswitch
mkdir -p "$OUT_ROOT"

NNODES=${SLURM_NNODES:-3}
TOTAL=$(( NNODES * 8 ))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29619
NVIDEOS=$(tail -n +2 "$CSV" | awk -F',' '{print $1}' | sort -u | grep -c .)
echo "[evtsw-all] nodes=$NNODES total_gpu=$TOTAL videos/ckpt=$NVIDEOS master=$MASTER_ADDR"

gen_ckpt() {  # $1 = checkpoint-N  OR  "step-0" (baseline, raw Base, no LoRA)
  local ckpt=$1
  local out="$OUT_ROOT/$ckpt"
  mkdir -p "$out"
  local have; have=$(ls "$out"/*.mp4 2>/dev/null | wc -l)
  if [ "$have" -ge "$NVIDEOS" ]; then echo "[evtsw-all] $ckpt already complete ($have/$NVIDEOS)"; return 0; fi
  # baseline = raw Helios-Base, no adapter; everything else loads the checkpoint's LoRA.
  local lora_arg="--lora_path $TRAIN_OUT/$ckpt"
  if [ "$ckpt" = "step-0" ]; then lora_arg=""; fi
  echo "[evtsw-all] === $ckpt : $have/$NVIDEOS done -> generating on $TOTAL GPU ==="
  srun --nodes="$NNODES" --ntasks="$TOTAL" --ntasks-per-node=8 --gres=gpu:h200:8 \
       --kill-on-bad-exit=1 bash -c '
    export PATH='"$PY_ENV"'/bin:$PATH
    export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
    export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
    export TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
    export RANK=$SLURM_PROCID WORLD_SIZE='"$TOTAL"' LOCAL_RANK=$SLURM_LOCALID
    export MASTER_ADDR='"$MASTER_ADDR"' MASTER_PORT='"$MASTER_PORT"'
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_evtswall_${SLURM_JOB_ID}_${SLURM_NODEID}
    export TMPDIR=/tmp/tmp_'"$USER"'_evtswall_${SLURM_JOB_ID}_${SLURM_NODEID}
    mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
    cd '"$REPO"'
    python infer_helios.py \
      --base_model_path '"$BASE_MODEL"' --transformer_path '"$BASE_MODEL"' \
      '"$lora_arg"' \
      --interactive_prompt_csv_path '"$CSV"' \
      --num_frames 1386 --fps 24 \
      --num_inference_steps 50 --guidance_scale 5.0 \
      --height 384 --width 640 \
      --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
      --output_folder '"$out"'
  ' || echo "[evtsw-all] $ckpt srun returned nonzero (continuing)"
  echo "[evtsw-all] $ckpt now: $(ls "$out"/*.mp4 2>/dev/null | wc -l)/$NVIDEOS"
}

# Sweep all current + newly-appearing checkpoints. Process ascending (so older steps come first);
# re-scan each pass to pick up new checkpoints from the running training. Stop when training is
# done (checkpoint-9900-final present) AND every checkpoint is complete.
while true; do
  # Greedily pick the NEWEST still-incomplete checkpoint each iteration, so a freshly-saved
  # checkpoint from the running training always jumps to the front of the queue.
  # Candidate order: newest checkpoint first, then step-0 (baseline) 2nd, then the rest descending.
  mapfile -t ckpts < <(ls -d "$TRAIN_OUT"/checkpoint-* 2>/dev/null | grep -oE 'checkpoint-[0-9]+' | sort -t- -k2 -nr)
  candidates=()
  [ ${#ckpts[@]} -gt 0 ] && candidates+=("${ckpts[0]}")
  candidates+=("step-0")
  [ ${#ckpts[@]} -gt 1 ] && candidates+=("${ckpts[@]:1}")
  next=""
  for ckpt in "${candidates[@]}"; do
    out="$OUT_ROOT/$ckpt"; have=$(ls "$out"/*.mp4 2>/dev/null | wc -l)
    if [ "$have" -lt "$NVIDEOS" ]; then next="$ckpt"; break; fi
  done
  if [ -n "$next" ]; then
    gen_ckpt "$next"        # do exactly one; loop re-scans -> newest-incomplete again
    continue
  fi
  # nothing incomplete right now
  if [ -d "$TRAIN_OUT/checkpoint-9900-final" ]; then
    echo "[evtsw-all] ALL checkpoints complete + training finished. EVTSW_ALL DONE $(date +%s)"; break
  fi
  echo "[evtsw-all] all current checkpoints done; sleeping 600s waiting for new ones"
  sleep 600
done

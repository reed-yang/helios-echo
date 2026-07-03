#!/bin/bash
#SBATCH --job-name=helios_evalq
#SBATCH --nodes=4
#SBATCH --nodelist=c-node03,c-node04,mc-node01,mc-node02   # freed full-FT nodes (LoRA 2538 keeps c-node05/06)
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/evalq_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/evalq_%j.out
# GLOBAL SINGLE-GPU WORK QUEUE so NO GPU is ever idle.
#   Each of the 32 GPUs runs an independent worker (world_size=1, NO NCCL) that atomically claims the next
#   pending per-video task (mv is atomic on beegfs), runs infer_helios for that ONE video, and loops until
#   the queue is empty. Long 2-min (group A) and short 1-min (group B) videos interleave -> natural load
#   balancing, every GPU busy until the tail.
#   Tasks (built once, on the batch node, below):
#     A__<step>__<i>     group A no-switch: qa/<i>.txt   -> 2904f t2v -> lora/step-<step>/<i>.mp4   (i=0..4)
#     B__<step>__<id>    group B evt-switch: qb/<id>.csv -> 1386f interp -> lora_evtsw/step-<step>/<id>.mp4
#   Group A already done for 500..3000 (valid, with --partial_path) -> only missing A re-queued.
#   Group B fully re-queued for ALL checkpoints (prompts were realigned to the 6-tag training format).
# Launch:  sbatch scripts/eval_live/eval_queue.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
TRAIN_DIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr
QA=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qa
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb
OUT_A=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr/lora
OUT_B=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr/lora_evtsw
QDIR=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr/.queue
NF_A=2904; NF_B=1386; STEPS=50; GUID=5.0; BIDS="1000 1001 1002 1003 1010"
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$OUT_A" "$OUT_B"

# ---- build the pending queue (batch node only, before any worker starts) ----
rm -rf "$QDIR"; mkdir -p "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"
# Group B: realigned prompts -> wipe stale outputs and re-queue every checkpoint.
for d in "$TRAIN_DIR"/checkpoint-*; do
  [ -f "$d/pytorch_lora_weights.safetensors" ] || continue
  step=$(basename "$d" | sed 's/checkpoint-//'); [[ "$step" =~ ^[0-9]+$ ]] || continue
  rm -rf "$OUT_B/step-$step"
  for id in $BIDS; do touch "$QDIR/pending/B__${step}__${id}"; done
  # Group A: only (re)queue items whose mp4 is missing (500..3000 already valid).
  for i in 0 1 2 3 4; do
    [ -f "$OUT_A/step-$step/$i.mp4" ] || touch "$QDIR/pending/A__${step}__${i}"
  done
done
NP=$(ls "$QDIR/pending" | wc -l)
echo "[evalq] $(date) queued $NP tasks across $(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l) nodes x 8 GPU"
ls "$QDIR/pending" | sort | sed 's/^/  /'

# ---- 32 independent single-GPU workers pulling from the queue ----
srun --ntasks=32 --ntasks-per-node=8 --gres=gpu:h200:8 --cpus-per-task=10 --exact --kill-on-bad-exit=0 bash -c '
  set -uo pipefail
  export PATH='"$PY_ENV"'/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  ME=$(hostname)-g$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_evalq_${ME} TMPDIR=/tmp/tmp_'"$USER"'_evalq_${ME}
  mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd '"$REPO"'
  QDIR='"$QDIR"'; EVAL_BASE='"$EVAL_BASE"'; TRAIN_DIR='"$TRAIN_DIR"'
  QA='"$QA"'; QB='"$QB"'; OUT_A='"$OUT_A"'; OUT_B='"$OUT_B"'
  NF_A='"$NF_A"'; NF_B='"$NF_B"'; STEPS='"$STEPS"'; GUID='"$GUID"'
  COMMON="--base_model_path $EVAL_BASE --transformer_path $EVAL_BASE --fps 24 --num_inference_steps $STEPS --guidance_scale $GUID --height 384 --width 640"
  n=0; empty=0
  while true; do
    claimed=""; task=""
    shopt -s nullglob
    for f in "$QDIR"/pending/*; do
      [ -e "$f" ] || continue
      bn=$(basename "$f")
      if mv "$f" "$QDIR/run/${ME}__${bn}" 2>/dev/null; then claimed="$QDIR/run/${ME}__${bn}"; task="$bn"; break; fi
    done
    shopt -u nullglob
    # Empty readdir is NOT a reliable end-of-queue signal: on beegfs the pending dir can briefly look empty
    # on a node that has not yet seen the batch-node creations, and at true end others may hold the last tasks.
    # Require a sustained-empty grace window (12 x 5s = 60s) before exiting, so no worker quits early.
    if [ -z "$task" ]; then
      empty=$((empty+1))
      if [ "$empty" -ge 12 ]; then echo "[$ME] queue empty 60s, ran $n tasks, exit $(date)"; break; fi
      sleep 5; continue
    fi
    empty=0
    grp=${task%%__*}; rest=${task#*__}; step=${rest%%__*}; item=${rest#*__}
    ckpt="$TRAIN_DIR/checkpoint-$step"; partial="$ckpt/transformer_partial.pth"
    echo "[$ME] $(date) START $task"
    if [ "$grp" = A ]; then
      tmp=$(mktemp -d "$TMPDIR/A_${step}_${item}_XXXX")
      python infer_helios.py $COMMON --lora_path "$ckpt" --partial_path "$partial" \
        --prompt_txt_path "$QA/$item.txt" --sample_type t2v --num_frames $NF_A --output_folder "$tmp"
      if [ -f "$tmp/0.mp4" ]; then mkdir -p "$OUT_A/step-$step"; mv -f "$tmp/0.mp4" "$OUT_A/step-$step/$item.mp4"; fi
      rm -rf "$tmp"
      ok=$([ -f "$OUT_A/step-$step/$item.mp4" ] && echo 1 || echo 0)
    else
      mkdir -p "$OUT_B/step-$step"
      python infer_helios.py $COMMON --lora_path "$ckpt" --partial_path "$partial" \
        --interactive_prompt_csv_path "$QB/$item.csv" --use_interpolate_prompt --interpolation_steps 0 \
        --interpolate_time 7 --num_frames $NF_B --output_folder "$OUT_B/step-$step"
      ok=$([ -f "$OUT_B/step-$step/$item.mp4" ] && echo 1 || echo 0)
    fi
    if [ "$ok" = 1 ]; then mv -f "$claimed" "$QDIR/done/${task}"; echo "[$ME] $(date) DONE $task";
    else mv -f "$claimed" "$QDIR/failed/${task}" 2>/dev/null; echo "[$ME] $(date) FAIL $task -> failed/"; fi
    n=$((n+1))
  done
'
echo "[evalq] $(date) all workers exited. done=$(ls "$QDIR/done" | wc -l) pending=$(ls "$QDIR/pending" 2>/dev/null | wc -l)"

#!/bin/bash
#SBATCH --job-name=helios_evalq2
#SBATCH --nodes=2
#SBATCH --nodelist=c-node05,c-node06          # freed LoRA nodes (full-SFT 2551 keeps c03/04+mc01/02)
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/evalq2_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/evalq2_%j.out
# GLOBAL SINGLE-GPU WORK QUEUE on 16 GPU (c-node05/06) inferring BOTH models so NO GPU idles.
#   16 independent workers (world_size=1, NO NCCL) atomically claim per-video tasks via `mv` under
#   eval_out_cfr/.queue2/{pending,run,done,failed}; sustained-60s-empty grace before a worker exits
#   (beegfs cross-node readdir lag must not kill workers early).
#   Task name = <model>_<grp>__<step>__<item> :
#     lora_A__<step>__<i>   LoRA t2v no-switch  : qa/<i>.txt   -> 2904f -> lora/step-<step>/<i>.mp4
#     lora_B__<step>__<id>  LoRA evt-switch      : qb/<id>.csv -> 1386f -> lora_evtsw/step-<step>/<id>.mp4
#     full_A__<step>__<i>   full-SFT t2v         : qa/<i>.txt   -> 2904f -> full/step-<step>/<i>.mp4
#     full_B__<step>__<id>  full-SFT evt-switch  : qb/<id>.csv -> 1386f -> full_evtsw/step-<step>/<id>.mp4
#   Both models load identically: base + --lora_path <ckpt> + --partial_path <ckpt>/transformer_partial.pth
#   (LoRA ckpt = r128, full-SFT ckpt = r256 + full patch_embedding; infer_helios auto-detects rank).
#   Scope: LoRA ckpts 4500..9000 (500..4000 already eval'd) + full-SFT ckpts 500,1000. Idempotent (skips
#   existing mp4). Launch:  sbatch scripts/eval_live/eval_queue2.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
LORA_DIR=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr
FULL_DIR=/mnt/beegfs/xiangbo/helios_runs/stage1_sft_cfr_v2
QA=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qa
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb
EOUT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr
QDIR=$EOUT/.queue2
NF_A=2904; NF_B=1386; STEPS=50; GUID=5.0; BIDS="1000 1001 1002 1003 1010"
# event-switch (group B) ONLY. LoRA sampled every 1000 steps (500..4000 already eval'd); full-SFT every ckpt.
LORA_STEPS="5000 6000 7000 8000 9000"
FULL_STEPS="500 1000"
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$EOUT"

# ---- build queue (batch node only) ----
rm -rf "$QDIR"; mkdir -p "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"
queue_model() {  # $1=model(lora|full) $2=train_dir $3=outA-name $4=outB-name $5=steps
  local m=$1 tdir=$2 oa=$3 ob=$4; shift 4
  for step in "$@"; do
    [ -f "$tdir/checkpoint-$step/pytorch_lora_weights.safetensors" ] || continue
    # event-switch (group B) only — group A (no-switch) intentionally skipped per user.
    for id in $BIDS; do
      [ -f "$EOUT/$ob/step-$step/$id.mp4" ] || touch "$QDIR/pending/${m}_B__${step}__${id}"
    done
  done
}
queue_model lora "$LORA_DIR" lora lora_evtsw $LORA_STEPS
queue_model full "$FULL_DIR" full full_evtsw $FULL_STEPS
NP=$(ls "$QDIR/pending" | wc -l)
echo "[evalq2] $(date) queued $NP tasks on c-node05/06 (16 GPU)"
ls "$QDIR/pending" | sort | sed 's/^/  /'

# ---- 16 single-GPU workers ----
srun --ntasks=16 --ntasks-per-node=8 --gres=gpu:h200:8 --cpus-per-task=10 --exact --kill-on-bad-exit=0 bash -c '
  set -uo pipefail
  export PATH='"$PY_ENV"'/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  ME=$(hostname)-g$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_evalq2_${ME} TMPDIR=/tmp/tmp_'"$USER"'_evalq2_${ME}
  mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd '"$REPO"'
  QDIR='"$QDIR"'; EVAL_BASE='"$EVAL_BASE"'; LORA_DIR='"$LORA_DIR"'; FULL_DIR='"$FULL_DIR"'
  QA='"$QA"'; QB='"$QB"'; EOUT='"$EOUT"'; NF_A='"$NF_A"'; NF_B='"$NF_B"'; STEPS='"$STEPS"'; GUID='"$GUID"'
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
    if [ -z "$task" ]; then
      empty=$((empty+1)); [ "$empty" -ge 12 ] && { echo "[$ME] queue empty 60s, ran $n, exit $(date)"; break; }
      sleep 5; continue
    fi
    empty=0
    head=${task%%__*}; model=${head%_*}; grp=${head#*_}
    rest=${task#*__}; step=${rest%%__*}; item=${rest#*__}
    if [ "$model" = lora ]; then tdir="$LORA_DIR"; oa="$EOUT/lora"; ob="$EOUT/lora_evtsw"; else tdir="$FULL_DIR"; oa="$EOUT/full"; ob="$EOUT/full_evtsw"; fi
    ckpt="$tdir/checkpoint-$step"; partial="$ckpt/transformer_partial.pth"
    echo "[$ME] $(date) START $task"
    if [ "$grp" = A ]; then
      tmp=$(mktemp -d "$TMPDIR/${model}_A_${step}_${item}_XXXX")
      python infer_helios.py $COMMON --lora_path "$ckpt" --partial_path "$partial" \
        --prompt_txt_path "$QA/$item.txt" --sample_type t2v --num_frames $NF_A --output_folder "$tmp"
      if [ -f "$tmp/0.mp4" ]; then mkdir -p "$oa/step-$step"; mv -f "$tmp/0.mp4" "$oa/step-$step/$item.mp4"; fi
      rm -rf "$tmp"
      ok=$([ -f "$oa/step-$step/$item.mp4" ] && echo 1 || echo 0)
    else
      mkdir -p "$ob/step-$step"
      python infer_helios.py $COMMON --lora_path "$ckpt" --partial_path "$partial" \
        --interactive_prompt_csv_path "$QB/$item.csv" --use_interpolate_prompt --interpolation_steps 0 \
        --interpolate_time 7 --num_frames $NF_B --output_folder "$ob/step-$step"
      ok=$([ -f "$ob/step-$step/$item.mp4" ] && echo 1 || echo 0)
    fi
    if [ "$ok" = 1 ]; then mv -f "$claimed" "$QDIR/done/${task}"; echo "[$ME] $(date) DONE $task";
    else mv -f "$claimed" "$QDIR/failed/${task}" 2>/dev/null; echo "[$ME] $(date) FAIL $task -> failed/"; fi
    n=$((n+1))
  done
'
echo "[evalq2] $(date) workers exited. done=$(ls "$QDIR/done"|wc -l) pending=$(ls "$QDIR/pending" 2>/dev/null|wc -l) failed=$(ls "$QDIR/failed"|wc -l)"

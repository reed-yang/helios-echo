#!/bin/bash
#SBATCH --job-name=helios_evallong
#SBATCH --nodes=2
#SBATCH --nodelist=c-node07,c-node08          # inference nodes (training is on c05/06)
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/evallong_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/evallong_%j.out
# LONG 3-min event-switch comparison across 4 model-configs x 3 new scenarios = 12 videos.
#   Each video: 19 events, interpolate_time 7 -> 133 chunks -> 4389 frames @24fps = ~3.05 min, switch ~every 9.6s.
#   Configs (task = <config>__<scenario>):
#     baseorig : Helios-Base (released transformer) + ORIGINAL prose prompts (long/orig/<sc>.csv)
#     basenew  : Helios-Base                         + NEW 6-tag prompts   (long/new/<sc>.csv)
#     lora9000 : eval_base_init(transformer_init) + lora_path stage1_lora_cfr/checkpoint-9000 + partial + NEW
#     sft6000  : eval_base_init(transformer_init) + lora_path stage1_sft_cfr_v2/checkpoint-6000 + partial + NEW
#   16 single-GPU workers (world_size=1, NO NCCL -> c07 safe), atomic mv claim, 60s-empty grace before exit.
#   Outputs: eval_out_cfr/long/<config>/<scenario>.mp4 . Idempotent (skip existing).
# Launch:  sbatch scripts/eval_live/eval_queue_long.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
SN=/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd
EVAL_BASE=/mnt/beegfs/xiangbo/helios_runs/eval_base_init
LORA=/mnt/beegfs/xiangbo/helios_runs/stage1_lora_cfr
SFT=/mnt/beegfs/xiangbo/helios_runs/stage1_sft_cfr_v2
LONG=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_long
OUT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr/long
QDIR=$OUT/.queue
NF=4389; IT=7; STEPS=50; GUID=5.0
CONFIGS="baseorig basenew lora9000 sft6000"; SCENS="2001 2002 2003"
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$OUT"

# ---- build queue (batch node) ----
rm -rf "$QDIR"; mkdir -p "$QDIR/pending" "$QDIR/run" "$QDIR/done" "$QDIR/failed"
for c in $CONFIGS; do for s in $SCENS; do
  [ -f "$OUT/$c/$s.mp4" ] || touch "$QDIR/pending/${c}__${s}"
done; done
NP=$(ls "$QDIR/pending" | wc -l)
echo "[evallong] $(date) queued $NP tasks (4 configs x 3 scenarios) on c-node07/08 (16 GPU)"
ls "$QDIR/pending" | sort | sed 's/^/  /'

# ---- 16 single-GPU workers ----
srun --ntasks=16 --ntasks-per-node=8 --gres=gpu:h200:8 --cpus-per-task=10 --exact --kill-on-bad-exit=0 bash -c '
  set -uo pipefail
  export PATH='"$PY_ENV"'/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  ME=$(hostname)-g$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_evallong_${ME} TMPDIR=/tmp/tmp_'"$USER"'_evallong_${ME}
  mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd '"$REPO"'
  QDIR='"$QDIR"'; SN='"$SN"'; EVAL_BASE='"$EVAL_BASE"'; LORA='"$LORA"'; SFT='"$SFT"'; LONG='"$LONG"'; OUT='"$OUT"'
  NF='"$NF"'; IT='"$IT"'; STEPS='"$STEPS"'; GUID='"$GUID"'
  COMMON="--fps 24 --num_inference_steps $STEPS --guidance_scale $GUID --height 384 --width 640 --use_interpolate_prompt --interpolation_steps 0 --interpolate_time $IT --num_frames $NF"
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
    cfg=${task%%__*}; sc=${task#*__}
    case "$cfg" in
      baseorig) MODEL="--base_model_path $SN --transformer_path $SN"; CSV="$LONG/orig/$sc.csv" ;;
      basenew)  MODEL="--base_model_path $SN --transformer_path $SN"; CSV="$LONG/new/$sc.csv" ;;
      lora9000) MODEL="--base_model_path $EVAL_BASE --transformer_path $EVAL_BASE --lora_path $LORA/checkpoint-9000 --partial_path $LORA/checkpoint-9000/transformer_partial.pth"; CSV="$LONG/new/$sc.csv" ;;
      sft6000)  MODEL="--base_model_path $EVAL_BASE --transformer_path $EVAL_BASE --lora_path $SFT/checkpoint-6000 --partial_path $SFT/checkpoint-6000/transformer_partial.pth"; CSV="$LONG/new/$sc.csv" ;;
      *) echo "[$ME] unknown config $cfg"; mv -f "$claimed" "$QDIR/failed/$task"; continue ;;
    esac
    outdir="$OUT/$cfg"; mkdir -p "$outdir"
    echo "[$ME] $(date) START $task"
    python infer_helios.py $MODEL $COMMON --interactive_prompt_csv_path "$CSV" --output_folder "$outdir"
    if [ -f "$outdir/$sc.mp4" ]; then mv -f "$claimed" "$QDIR/done/$task"; echo "[$ME] $(date) DONE $task";
    else mv -f "$claimed" "$QDIR/failed/$task" 2>/dev/null; echo "[$ME] $(date) FAIL $task -> failed/"; fi
    n=$((n+1))
  done
'
echo "[evallong] $(date) workers exited. done=$(ls "$QDIR/done"|wc -l) pending=$(ls "$QDIR/pending" 2>/dev/null|wc -l) failed=$(ls "$QDIR/failed"|wc -l)"

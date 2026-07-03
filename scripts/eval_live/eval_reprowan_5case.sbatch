#!/bin/bash
#SBATCH --job-name=helios_rw5case
#SBATCH --nodes=1
#SBATCH --nodelist=c-node07          # run on freed inference node after the long-inference job ends
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:h200:5
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/rw5case_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/rw5case_%j.out
# Sanity-check inference of the repro-wan (Stage-1-from-raw-Wan) run at a given CHECKPOINT, on the original
# 5 one-minute event-switch cases (ids 1000/1001/1002/1003/1010, aligned 6-tag prompts, 6 events x
# interpolate_time 7 = 42 chunks = 1386 frames). One id per GPU (5 GPUs).
# Repro-wan loads from RAW WAN weights, so the transformer is built from a custom dir = Wan weights +
# Helios config (repro_wan_infer_base) so the multi-term-memory patches exist; then lora + partial apply.
# Pass CKPT=<step> (default 1000). Launch:  CKPT=1000 sbatch scripts/eval_live/eval_reprowan_5case.sbatch
set -uo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
SN=/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd
INFER_TF=/mnt/beegfs/xiangbo/helios_runs/eval_base_init   # transformer_init: Wan-valued base + materialized memory patches
CKPT=${CKPT:-1000}
CKDIR=/mnt/beegfs/xiangbo/helios_runs/stage1_repro_wan/checkpoint-$CKPT
QB=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/qb            # per-id 6-tag CSVs (1000..1010)
OUT=/mnt/beegfs/xiangbo/helios_runs/eval_out_cfr/reprowan_step${CKPT}
IDS=(1000 1001 1002 1003 1010)
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs "$OUT"
echo "[rw5case] $(date) checkpoint=$CKDIR out=$OUT"

srun --ntasks=5 --ntasks-per-node=5 --gres=gpu:h200:5 --cpus-per-task=8 --exact --kill-on-bad-exit=0 bash -c '
  set -uo pipefail
  export PATH='"$PY_ENV"'/bin:$PATH HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface PYTHONPATH='"$REPO"':${PYTHONPATH:-}
  export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled NCCL_P2P_LEVEL=NVL
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  ME=$(hostname)-g$SLURM_LOCALID
  export TRITON_CACHE_DIR=/tmp/triton_'"$USER"'_rw5_${ME} TMPDIR=/tmp/tmp_'"$USER"'_rw5_${ME}
  mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"; cd '"$REPO"'
  IDS=(1000 1001 1002 1003 1010); id=${IDS[$SLURM_LOCALID]}
  SN='"$SN"'; INFER_TF='"$INFER_TF"'; CKDIR='"$CKDIR"'; QB='"$QB"'; OUT='"$OUT"'
  echo "[$ME] $(date) START id=$id"
  python infer_helios.py --base_model_path "$SN" --transformer_path "$INFER_TF" \
    --lora_path "$CKDIR" --partial_path "$CKDIR/transformer_partial.pth" \
    --interactive_prompt_csv_path "$QB/$id.csv" --use_interpolate_prompt --interpolation_steps 0 --interpolate_time 7 \
    --num_frames 1386 --num_inference_steps 50 --guidance_scale 5.0 --height 384 --width 640 --fps 24 \
    --output_folder "$OUT"
  echo "[$ME] $(date) DONE id=$id ok=$([ -f "$OUT/$id.mp4" ] && echo 1 || echo 0)"
'
echo "[rw5case] $(date) finished. videos: $(ls "$OUT"/*.mp4 2>/dev/null | wc -l)/5"

#!/bin/bash
#SBATCH --job-name=helios_eval_me1
#SBATCH --nodes=1
#SBATCH --nodelist=c-node03          # freed after training; c-node08 still held by another job
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/eval_me1_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/eval_me1_%j.out
# One-shot multi-event (prompt-switching) eval: generate 15 test + 15 train switch-videos for
# the BASELINE (raw Helios-Base, no LoRA) and the FINAL checkpoint, then HeliosBench. The cluster
# was saturated during training so live per-checkpoint eval wasn't possible; this runs at the end.
# Judge mainly by EYE: does the content visibly switch at each annotated boundary?
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

export REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
export PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PATH=$PY_ENV/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}

export EVAL_PROMPT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_multievent
export EVAL_OUT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_out_multievent
# Local Helios-Base snapshot dir for base_model + transformer (avoids the multi-rank HF-Hub race).
export BASE_MODEL=/mnt/beegfs/xiangbo/.cache/huggingface/hub/models--BestWishYsh--Helios-Base/snapshots/21cf91da9cae0270a9405000c2e798f83770b1dd
# c-node03 has no hot GPU1 -> use all 8 GPUs.
export EVAL_GPUS=0,1,2,3,4,5,6,7
export NUM_STEPS=50 GUIDANCE=5.0 RUN_METRICS=1

CKPT_FINAL=/mnt/beegfs/xiangbo/helios_runs/stage1_multievent_ohv/checkpoint-1050-final

# Build the fixed 15+15 prompt sets (idempotent).
python "$REPO/scripts/eval_live/build_eval_prompts_multievent.py" --n-test 15 --n-train 15 --out-dir "$EVAL_PROMPT_DIR"

echo "===== BASELINE (raw Helios-Base, no LoRA) ====="
bash "$REPO/scripts/eval_live/run_eval_once_multievent.sh" base 0 both

echo "===== FINAL checkpoint (step 1050) ====="
bash "$REPO/scripts/eval_live/run_eval_once_multievent.sh" "$CKPT_FINAL" 1050 both

echo "EVAL_ME_ONESHOT DONE $(date +%s)"

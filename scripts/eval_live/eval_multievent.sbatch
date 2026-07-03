#!/bin/bash
#SBATCH --job-name=helios_eval_me
#SBATCH --nodes=1
#SBATCH --nodelist=c-node08          # node8 is the designated INFERENCE node (hot GPU1 -> not for training)
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00             # long-running watcher; must outlive training
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/eval_me_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/eval_me_%j.out
# Live multi-event (prompt-switching) eval node: watches the training checkpoints and, for the
# newest, generates 15 test + 15 train switch-videos and runs HeliosBench. Judge mainly by eye.
# Launch:  sbatch scripts/eval_live/eval_multievent.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

export REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
export PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PATH=$PY_ENV/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}

export TRAIN_OUTPUT_DIR=/mnt/beegfs/xiangbo/helios_runs/stage1_multievent_ohv
export EVAL_PROMPT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_multievent
export EVAL_OUT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_out_multievent
export STATE_FILE=$EVAL_OUT_DIR/.last_evaluated_step
export EVAL_RUN_SCRIPT=run_eval_once_multievent.sh
# c-node08 GPU1 thermally throttles (94C / 345MHz / SW_THERMAL_SLOWDOWN) and wedges its rank.
# Skip GPU1; run on the 7 healthy GPUs.
export EVAL_GPUS=0,2,3,4,5,6,7
export NUM_STEPS=50 GUIDANCE=5.0
export RUN_METRICS=1            # reference-free HeliosBench; manual inspection is the primary signal
export RUN_BASELINE=1           # eval raw Helios-Base (no LoRA) once as step 0
export POLL_INTERVAL=120 WHICH=both

# Build the fixed multi-event prompt sets once (idempotent): 15 test + 15 train.
python "$REPO/scripts/eval_live/build_eval_prompts_multievent.py" \
  --n-test 15 --n-train 15 --out-dir "$EVAL_PROMPT_DIR"

bash "$REPO/scripts/eval_live/watch_and_eval.sh"

#!/bin/bash
#SBATCH --job-name=helios_eval
#SBATCH --nodes=1
#SBATCH --nodelist=c-node08          # node8 is the designated INFERENCE node (hot GPU1 -> not for training)
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00             # long-running watcher; must outlive training (default cap was 1h)
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/eval_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/eval_%j.out
# Live eval node: watches the training checkpoints and evaluates the newest one.
# Launch:  sbatch scripts/eval_live/eval.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

export REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
export PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PATH=$PY_ENV/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}

export TRAIN_OUTPUT_DIR=/mnt/beegfs/xiangbo/helios_runs/stage1_continue_ohv
export EVAL_PROMPT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_prompts
export EVAL_OUT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_out
# ROOT CAUSE of eval wedges: c-node08 GPU1 thermally throttles (94C, SM clock 345MHz vs
# ~1500MHz, SW_THERMAL_SLOWDOWN) -> its rank crawls and blocks the whole data-parallel round.
# Skip GPU1 entirely; run on the 7 healthy GPUs. (This is the "node8 GPU1 runs hot" the user warned about.)
export EVAL_GPUS=0,2,3,4,5,6,7
# 330 frames is fine on healthy GPUs — the earlier "long-gen hang" was actually GPU1 throttling, not length.
export NUM_FRAMES=330 NUM_STEPS=50 GUIDANCE=5.0
export RUN_METRICS=1            # set 0 to generate only; naturalness off unless NATURALNESS_API_KEY set
export RUN_BASELINE=1           # eval raw Helios-Base (no LoRA) once as step 0
export POLL_INTERVAL=120 WHICH=both

# Build the fixed prompt sets once (idempotent): 15 test + 15 train, version=medium.
python "$REPO/scripts/eval_live/build_eval_prompts.py" --version medium --num-frames "$NUM_FRAMES" \
  --n-test 15 --n-train 15 --out-dir "$EVAL_PROMPT_DIR"

bash "$REPO/scripts/eval_live/watch_and_eval.sh"

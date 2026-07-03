#!/bin/bash
#SBATCH --job-name=helios_eval24
#SBATCH --nodes=3
#SBATCH --nodelist=c-node03,c-node05,c-node06   # 3 idle nodes = 24 GPUs, multi-node data-parallel eval
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00             # long-running watcher; must outlive training
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/eval24_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/eval24_%j.out
# Live eval node for the video_single_24FPS run: watches the training checkpoints and
# evaluates the newest one. 20 test + 20 fixed-random train, SHORT caption, each generated
# as a LONG ~1.5 min video (2178 frames = 33*66 chunks @24fps ~= 91s) to stress long-gen,
# independent of the source clip length.
# Launch:  sbatch scripts/eval_live/eval_video_single.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

export REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
export PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PATH=$PY_ENV/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}

export TRAIN_OUTPUT_DIR=/mnt/beegfs/xiangbo/helios_runs/stage1_video_single_24fps
# Fresh dirs for the long-gen eval (keeps the earlier 330f/100-video results intact).
export EVAL_PROMPT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_vs24_long
export EVAL_OUT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_out_vs24_long
export STATE_FILE=$EVAL_OUT_DIR/.last_evaluated_step
# MULTI-NODE data-parallel eval across the 3 idle nodes (24 GPUs). gen_one fans infer_helios
# over all 24 via srun env-launcher (data-parallel by prompt index). Metrics run single-node on head.
export EVAL_NNODES=${SLURM_NNODES:-3}   # derive from the actual allocation (2 or 3 nodes)
export EVAL_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export EVAL_MASTER_PORT=29615
# LONG generation: 2178 frames = 33*66 chunks @24fps ~= 91s (1-2 min), fixed regardless of clip length.
export NUM_FRAMES=2178 NUM_STEPS=50 GUIDANCE=5.0
export RUN_METRICS=1            # HeliosBench reference-free metrics (naturalness off unless API key set)
# Baseline already ~done on the old node (34/40 in eval_out_vs24_long/step-0); skip it and go
# straight to the latest trained checkpoint so trained long videos appear ~3x faster on 24 GPUs.
export RUN_BASELINE=0
export POLL_INTERVAL=120 WHICH=both
export EVAL_RUN_SCRIPT=run_eval_once.sh

# Build the fixed prompt sets once (idempotent): 20 test + 20 train, version=short, long num_frames.
python "$REPO/scripts/eval_live/build_eval_prompts_video_single.py" --version short --num-frames "$NUM_FRAMES" \
  --n-test 20 --n-train 20 --out-dir "$EVAL_PROMPT_DIR"

bash "$REPO/scripts/eval_live/watch_and_eval.sh"

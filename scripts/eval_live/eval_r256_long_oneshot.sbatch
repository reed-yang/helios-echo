#!/bin/bash
#SBATCH --job-name=helios_r256long
#SBATCH --nodes=1
#SBATCH --nodelist=c-node08          # the only idle node; GPU1 throttles -> skipped via EVAL_GPUS
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/eval_r256long_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/eval_r256long_%j.out
# ONE-SHOT long-gen inference of the previously-trained r256 single-prompt LoRA
# (stage1_continue_ohv_r256_warm/checkpoint-900-final). Same 20 test + 20 train SHORT-caption
# prompts and 2178-frame (~91s) length as the vs24 long eval, for direct comparison.
# Single-node (c-node08, 7 GPUs skipping the hot GPU1). Launch: sbatch scripts/eval_live/eval_r256_long_oneshot.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs

export REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
export PY_ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
export HF_HOME=/mnt/beegfs/xiangbo/.cache/huggingface
export PATH=$PY_ENV/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_${USER}_r256long_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${USER}_r256long_${SLURM_JOB_ID}

R256_CKPT=/mnt/beegfs/xiangbo/helios_runs/stage1_continue_ohv_r256_warm/checkpoint-900-final
export BASE_MODEL=BestWishYsh/Helios-Base
export EVAL_PROMPT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_prompts_vs24_long   # reuse the built 20+20 long prompts
export EVAL_OUT_DIR=/mnt/beegfs/xiangbo/helios_runs/eval_out_r256_long
export EVAL_GPUS=0,2,3,4,5,6,7        # skip c-node08 GPU1 (thermal throttle)
export NUM_FRAMES=2178 NUM_STEPS=50 GUIDANCE=5.0 H=384 W=640
export RUN_METRICS=1
# single-node torchrun path (EVAL_NNODES unset)

# Ensure the long prompt sets exist (idempotent; builds 20 test + 20 train, short caption).
python "$REPO/scripts/eval_live/build_eval_prompts_video_single.py" --version short --num-frames "$NUM_FRAMES" \
  --n-test 20 --n-train 20 --out-dir "$EVAL_PROMPT_DIR"

# One eval round over the r256 checkpoint (STEP label "r256" -> step-r256/{test,train}).
bash "$REPO/scripts/eval_live/run_eval_once.sh" "$R256_CKPT" r256 both
echo "R256_LONG DONE $(date +%s)"

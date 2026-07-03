#!/bin/bash
#SBATCH --job-name=helios_evtswW
#SBATCH --nodes=3
#SBATCH --nodelist=c-node03,c-node05,c-node06   # the 24 non-training GPUs
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/mnt/beegfs/xiangbo/helios_runs/logs/evtswW_%j.out
#SBATCH --error=/mnt/beegfs/xiangbo/helios_runs/logs/evtswW_%j.out
# Event-switching long inference over EVERY vs24 checkpoint (24-GPU data-parallel sweep + watch).
# Launch:  sbatch scripts/eval_live/eval_vs24_eventswitch_watch.sbatch
set -euo pipefail
mkdir -p /mnt/beegfs/xiangbo/helios_runs/logs
export REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
export PATH=/mnt/beegfs/yuheng/miniconda3/envs/helios/bin:$PATH
echo "evtswW: nodes=$SLURM_JOB_NODELIST"
bash "$REPO/scripts/eval_live/run_eventswitch_all.sh"

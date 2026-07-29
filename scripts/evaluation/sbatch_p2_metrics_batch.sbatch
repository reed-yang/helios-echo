#!/bin/bash
#SBATCH -p gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --job-name=p2_metrics
#SBATCH --output=/mnt/beegfs/siyuan/workspace/helios-echo/results/p2_interim/logs/%x_%j.log

# CPU-only batch metrics: compute timeseries metrics for every video of the
# given campaigns that lacks a JSON. Meant to run via Slurm dependency after
# the inference jobs finish, so the whole chain survives session loss.
# Usage: sbatch sbatch_p2_metrics_batch.sbatch <run_id> [<run_id> ...]
set -uo pipefail
cd /mnt/beegfs/siyuan/workspace/helios-echo
ENV=/mnt/beegfs/yuheng/miniconda3/envs/helios
M=tools/long_video_eval/scripts/run_helios_long_timeseries_metric.py

total=0; done_ct=0
for rid in "$@"; do
  for mani in results/p2_interim/$rid/*/prompt_*.manifest.json; do
    [ -f "$mani" ] || continue
    dir=$(dirname "$mani"); arm=$(basename "$dir")
    stem=$(basename "$mani" .manifest.json)
    out="results/p2_interim/$rid/metrics/${arm}_${stem}.json"
    total=$((total+1))
    if [ ! -f "$out" ]; then
      mkdir -p "results/p2_interim/$rid/metrics"
      PYTHONPATH=. "$ENV/bin/python" $M "$dir/$stem.mp4" --out "$out" \
        --chunk_frames 33 --motion_height 384 --motion_width 640 \
        > "results/p2_interim/$rid/metrics/${arm}_${stem}.log" 2>&1
    fi
    [ -f "$out" ] && done_ct=$((done_ct+1))
  done
done
echo "METRICS_BATCH done=$done_ct/$total"
# Refresh the preview site so new campaigns/metrics appear automatically.
"$ENV/bin/python" scripts/evaluation/build_p2_site.py || true
echo "EXIT_CODE=0"

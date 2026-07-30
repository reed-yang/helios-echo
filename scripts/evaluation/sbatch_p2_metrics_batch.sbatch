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

# One 349-chunk (8-minute) video costs 13-18 min single-threaded, so a
# sequential batch of a whole campaign set runs for hours. Each metric is
# single-threaded and the job holds 16 CPUs, so run several at a time. Missing
# JSONs are recomputed and existing ones skipped, which makes the batch
# idempotent: it is safe to cancel and resubmit without losing finished work.
PARALLEL="${P2_METRICS_PARALLEL:-4}"
cmds=$(mktemp)
trap 'rm -f "$cmds"' EXIT

total=0; queued=0
for rid in "$@"; do
  for mani in results/p2_interim/$rid/*/prompt_*.manifest.json; do
    [ -f "$mani" ] || continue
    dir=$(dirname "$mani"); arm=$(basename "$dir")
    stem=$(basename "$mani" .manifest.json)
    out="results/p2_interim/$rid/metrics/${arm}_${stem}.json"
    total=$((total+1))
    [ -f "$out" ] && continue
    mkdir -p "results/p2_interim/$rid/metrics"
    printf 'PYTHONPATH=. %s/bin/python %s %s --out %s --chunk_frames 33 --motion_height 384 --motion_width 640 > %s 2>&1\n' \
      "$ENV" "$M" "$dir/$stem.mp4" "$out" "results/p2_interim/$rid/metrics/${arm}_${stem}.log" >> "$cmds"
    queued=$((queued+1))
  done
done
echo "METRICS_BATCH queued=$queued/$total parallel=$PARALLEL"
if [ "$queued" -gt 0 ]; then
  xargs -P "$PARALLEL" -I CMD bash -c CMD < "$cmds"
fi

done_ct=0
for rid in "$@"; do
  for mani in results/p2_interim/$rid/*/prompt_*.manifest.json; do
    [ -f "$mani" ] || continue
    arm=$(basename "$(dirname "$mani")")
    stem=$(basename "$mani" .manifest.json)
    [ -f "results/p2_interim/$rid/metrics/${arm}_${stem}.json" ] && done_ct=$((done_ct+1))
  done
done
echo "METRICS_BATCH done=$done_ct/$total"
# Refresh the preview site so new campaigns/metrics appear automatically.
"$ENV/bin/python" scripts/evaluation/build_p2_site.py || true
echo "EXIT_CODE=0"

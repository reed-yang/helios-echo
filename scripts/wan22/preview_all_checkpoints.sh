#!/bin/bash
# Eval EVERY 500-step checkpoint (not just the latest) with the 5-case 1-minute spec.
# Merges each checkpoint inline (big-RAM shell node) and queues a t2v infer on mc-node02/c-node08.
# Idempotent per step (preview_infer_step.sh skips already-inferred/queued). Safe to re-run.
#
#   bash scripts/wan22/preview_all_checkpoints.sh [CKPT_ROOT] [BASE] [PREFIX]
#   init (default): preview_all_checkpoints.sh
#   post: preview_all_checkpoints.sh /mnt/.../stage1_post_wan22full /mnt/.../stage1_init_wan22full_merged post
set -euo pipefail
REPO=/mnt/beegfs/xiangbo/Folder/Research/2026/interactive_streaming_model/helios-team
RUNS=/mnt/beegfs/xiangbo/helios_runs
CKPT_ROOT=${1:-$RUNS/stage1_init_wan22full}
BASE=${2:-/mnt/beegfs/xiangbo/wan_diffusers/Wan2.2-TI2V-5B-Diffusers}
PREFIX=${3:-init}
cd "$REPO"
# ascending step order so earlier checkpoints eval first
for ck in $(ls -d "$CKPT_ROOT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | grep -vE 'final|[^0-9]' | sort -n); do
  echo "=== preview $PREFIX checkpoint-$ck ==="
  bash scripts/wan22/preview_infer_step.sh "$ck" "$CKPT_ROOT" "$BASE" "$PREFIX" || echo "  (step $ck failed, continuing)"
done
echo "preview_all_checkpoints DONE for $PREFIX"

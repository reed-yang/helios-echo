#!/usr/bin/env bash
# Sharded full/start15/end15 evaluation for long Helios videos.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-tmp/outputs/helios_60s90s_prompt_sets/60s/long100_original}"
SEGMENT_ROOT="${SEGMENT_ROOT:-${SOURCE_ROOT}_ratio15_segments}"
SHARD_ROOT="${SHARD_ROOT:-${SEGMENT_ROOT}_shards}"
OUTPUT_ROOT="${OUTPUT_ROOT:-tmp/outputs/evaluation-results/helios_60s90s_prompt_sets}"
if [ -z "${OUTPUT_PREFIX:-}" ]; then
    parent="$(basename "$(dirname "$SOURCE_ROOT")")"
    child="$(basename "$SOURCE_ROOT")"
    OUTPUT_PREFIX="${parent}_${child}_ratio15"
fi

PROMPT_FILE="${PROMPT_FILE:-}"
MODELS="${MODELS:-base,distilled}"
SEGMENTS="${SEGMENTS:-full,start15,end15}"
METRICS="${METRICS:-dover,pickscore,hpsv3}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUN_QUALITY_METRICS="${RUN_QUALITY_METRICS:-1}"
RUN_VBENCH="${RUN_VBENCH:-1}"
SKIP_DONE_METRICS="${SKIP_DONE_METRICS:-1}"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON="${PYTHON:-python}"

FPS="${FPS:-24}"
DRIFT_RATIO="${DRIFT_RATIO:-0.15}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
OVERWRITE_SEGMENTS="${OVERWRITE_SEGMENTS:-0}"
RESET_SHARDS="${RESET_SHARDS:-1}"

# Preference metrics are expensive on 60s/90s videos. Default to one frame per
# 33-frame Helios chunk; set FRAMES_PER_CHUNK=4 for the denser 10s protocol.
FRAME_CHUNK_SIZE="${FRAME_CHUNK_SIZE:-33}"
FRAMES_PER_CHUNK="${FRAMES_PER_CHUNK:-1}"
FRAME_SAMPLES="${FRAME_SAMPLES:-8}"
HPSV3_BATCH_SIZE="${HPSV3_BATCH_SIZE:-1}"

EVAL_CKPT_ROOT="${EVAL_CKPT_ROOT:-$PROJECT_ROOT/tmp/models/eval_ckpts}"
DOVER_OPT="${DOVER_OPT:-$PROJECT_ROOT/third-party/DOVER/dover.yml}"
DOVER_WEIGHTS="${DOVER_WEIGHTS:-$EVAL_CKPT_ROOT/DOVER/DOVER.pth}"
HPSV3_CHECKPOINT="${HPSV3_CHECKPOINT:-$EVAL_CKPT_ROOT/HPSv3/HPSv3.safetensors}"
HPSV3_QWEN_MODEL_PATH="${HPSV3_QWEN_MODEL_PATH:-$EVAL_CKPT_ROOT/Qwen2-VL-7B-Instruct}"
PICKSCORE_PROCESSOR="${PICKSCORE_PROCESSOR:-$EVAL_CKPT_ROOT/PickScore/CLIP-ViT-H-14-laion2B-s32B-b79K}"
PICKSCORE_MODEL="${PICKSCORE_MODEL:-$EVAL_CKPT_ROOT/PickScore/PickScore_v1}"

VBENCH_ROOT="${VBENCH_ROOT:-$PROJECT_ROOT/third-party/FastVideo/fastvideo/third_party/eval/vbench}"
VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$EVAL_CKPT_ROOT/VBench}"
VBENCH_DIMS="${VBENCH_DIMS:-subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality}"
VBENCH_LOCAL="${VBENCH_LOCAL:-1}"
VBENCH_MASTER_PORT_BASE="${VBENCH_MASTER_PORT_BASE:-29600}"

split_list() {
    local value="$1"
    value="${value//,/ }"
    # shellcheck disable=SC2206
    SPLIT_RESULT=($value)
}

require_path() {
    local path="$1"
    local label="$2"
    if [ ! -e "$path" ]; then
        echo "ERROR: missing $label: $path" >&2
        exit 1
    fi
}

metric_enabled() {
    local needle="$1"
    for metric in "${METRIC_LIST[@]}"; do
        if [ "$metric" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

if [ -z "$PROMPT_FILE" ]; then
    PROMPT_FILE="$SOURCE_ROOT/prompts.txt"
fi

require_path "$SOURCE_ROOT" "source root"
require_path "$PROMPT_FILE" "prompt file"

if [ -n "$CONDA_ENV" ]; then
    require_path "$CONDA_SH" "conda.sh"
    set +u
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    set -u
fi

split_list "$MODELS"; MODEL_LIST=("${SPLIT_RESULT[@]}")
split_list "$SEGMENTS"; SEGMENT_LIST=("${SPLIT_RESULT[@]}")
split_list "$METRICS"; METRIC_LIST=("${SPLIT_RESULT[@]}")
split_list "$GPUS"; GPU_LIST=("${SPLIT_RESULT[@]}")
NUM_SHARDS="${#GPU_LIST[@]}"
if [ "$NUM_SHARDS" -le 0 ]; then
    echo "ERROR: GPUS resolved to an empty list" >&2
    exit 1
fi

if [ "$RUN_QUALITY_METRICS" = "1" ]; then
    metric_enabled dover && require_path "$DOVER_OPT" "DOVER config"
    metric_enabled dover && require_path "$DOVER_WEIGHTS" "DOVER weights"
    metric_enabled hpsv3 && require_path "$HPSV3_CHECKPOINT" "HPSv3 weights"
    metric_enabled hpsv3 && require_path "$HPSV3_QWEN_MODEL_PATH" "Qwen2-VL-7B snapshot"
    metric_enabled pickscore && require_path "$PICKSCORE_PROCESSOR" "PickScore processor"
    metric_enabled pickscore && require_path "$PICKSCORE_MODEL" "PickScore model"
fi
if [ "$RUN_VBENCH" = "1" ]; then
    require_path "$VBENCH_ROOT/evaluate.py" "VBench evaluate.py"
fi

export PROJECT_ROOT
export HF_HOME="${HF_HOME:-$EVAL_CKPT_ROOT/.hf_cache}"
export HPSV3_CHECKPOINT
export HPSV3_QWEN_MODEL_PATH
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/tmp/matplotlib}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export VBENCH_CACHE_DIR
export TORCH_HOME="${TORCH_HOME:-$VBENCH_CACHE_DIR/torch}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third-party/DOVER:$PROJECT_ROOT/third-party/HPSv3:$PROJECT_ROOT/third-party/PickScore:$VBENCH_ROOT:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_ROOT" "$MPLCONFIGDIR"

cat <<EOF
[ratio-eval] project_root       : $PROJECT_ROOT
[ratio-eval] source_root        : $SOURCE_ROOT
[ratio-eval] segment_root       : $SEGMENT_ROOT
[ratio-eval] shard_root         : $SHARD_ROOT
[ratio-eval] output_root        : $OUTPUT_ROOT
[ratio-eval] output_prefix      : $OUTPUT_PREFIX
[ratio-eval] prompt_file        : $PROMPT_FILE
[ratio-eval] models             : ${MODEL_LIST[*]}
[ratio-eval] segments           : ${SEGMENT_LIST[*]}
[ratio-eval] metrics            : ${METRIC_LIST[*]}
[ratio-eval] vbench dims        : $VBENCH_DIMS
[ratio-eval] gpus               : ${GPU_LIST[*]}
[ratio-eval] drift ratio        : $DRIFT_RATIO
[ratio-eval] frame sampling     : chunk=$FRAME_CHUNK_SIZE frames_per_chunk=$FRAMES_PER_CHUNK
[ratio-eval] conda env          : ${CONDA_DEFAULT_ENV:-}
[ratio-eval] python             : $(command -v "$PYTHON")
EOF

prepare_args=(
    --source-root "$SOURCE_ROOT"
    --output-root "$SEGMENT_ROOT"
    --prompt-file "$PROMPT_FILE"
    --models "$MODELS"
    --fps "$FPS"
    --ratio "$DRIFT_RATIO"
)
if [ "$MAX_VIDEOS" -gt 0 ]; then
    prepare_args+=(--max-videos "$MAX_VIDEOS")
fi
if [ "$OVERWRITE_SEGMENTS" = "1" ]; then
    prepare_args+=(--overwrite)
fi

echo "[ratio-eval] preparing full/start15/end15 segments"
"$PYTHON" scripts/evaluation/prepare_helios_ratio_segments.py "${prepare_args[@]}"

if [ "$RESET_SHARDS" = "1" ]; then
    rm -rf "$SHARD_ROOT"
fi
mkdir -p "$SHARD_ROOT"

echo "[ratio-eval] building shard video roots"
for segment in "${SEGMENT_LIST[@]}"; do
    for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
        for model in "${MODEL_LIST[@]}"; do
            mkdir -p "$SHARD_ROOT/$segment/shard_$(printf '%02d' "$shard_idx")/$model/videos"
        done
    done

    for model in "${MODEL_LIST[@]}"; do
        src_dir="$SEGMENT_ROOT/$segment/$model/videos"
        require_path "$src_dir" "$segment/$model videos"
        for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
            dst_dir="$SHARD_ROOT/$segment/shard_$(printf '%02d' "$shard_idx")/$model/videos"
            if [ -f "$src_dir/scores.json" ]; then
                ln -sfn "$(readlink -f "$src_dir/scores.json")" "$dst_dir/scores.json"
            fi
            if [ -f "$src_dir/vbench_prompts.json" ]; then
                ln -sfn "$(readlink -f "$src_dir/vbench_prompts.json")" "$dst_dir/vbench_prompts.json"
            fi
        done
        shopt -s nullglob
        for video in "$src_dir"/*.mp4; do
            stem="$(basename "$video" .mp4)"
            idx=$((10#$stem))
            shard_idx=$((idx % NUM_SHARDS))
            dst_dir="$SHARD_ROOT/$segment/shard_$(printf '%02d' "$shard_idx")/$model/videos"
            ln -sfn "$(readlink -f "$video")" "$dst_dir/$(basename "$video")"
        done
        shopt -u nullglob
    done
done

run_metric_shards() {
    local segment="$1"
    local metric="$2"
    echo "[ratio-eval] metric=$metric segment=$segment"
    mkdir -p "$OUTPUT_ROOT/logs/${OUTPUT_PREFIX}_${segment}"
    local pids=()
    for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
        local gpu="${GPU_LIST[$shard_idx]}"
        local shard_name
        shard_name="shard_$(printf '%02d' "$shard_idx")"
        (
            set -euo pipefail
            export CUDA_VISIBLE_DEVICES="$gpu"
            "$PYTHON" scripts/evaluation/run_helios_long_segment_metric.py \
                --metric "$metric" \
                --video-root "$SHARD_ROOT/$segment/$shard_name" \
                --output-root "$OUTPUT_ROOT" \
                --output-name "${OUTPUT_PREFIX}_${segment}_${shard_name}" \
                --prompt-file "$PROMPT_FILE" \
                --run-glob '*/videos' \
                --device cuda:0 \
                --frame-samples "$FRAME_SAMPLES" \
                --frame-chunk-size "$FRAME_CHUNK_SIZE" \
                --frames-per-chunk "$FRAMES_PER_CHUNK" \
                --max-videos "$MAX_VIDEOS" \
                --pickscore-processor "$PICKSCORE_PROCESSOR" \
                --pickscore-model "$PICKSCORE_MODEL" \
                --hpsv3-checkpoint "$HPSV3_CHECKPOINT" \
                --hpsv3-batch-size "$HPSV3_BATCH_SIZE" \
                --dover-opt "$DOVER_OPT" \
                --dover-weights "$DOVER_WEIGHTS"
        ) >"$OUTPUT_ROOT/logs/${OUTPUT_PREFIX}_${segment}/${metric}_${shard_name}.log" 2>&1 &
        pids+=("$!")
        echo "[ratio-eval]   launched $metric $segment $shard_name on gpu $gpu pid=$!"
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "ERROR: one or more shards failed for metric=$metric segment=$segment" >&2
        echo "Check logs in $OUTPUT_ROOT/logs/${OUTPUT_PREFIX}_${segment}" >&2
        exit 1
    fi
}

metric_done_for_segment() {
    local segment="$1"
    local metric="$2"
    "$PYTHON" - "$OUTPUT_ROOT" "$OUTPUT_PREFIX" "$segment" "$metric" "$NUM_SHARDS" <<'PY'
import json
import pathlib
import sys

output_root = pathlib.Path(sys.argv[1])
output_prefix = sys.argv[2]
segment = sys.argv[3]
metric = sys.argv[4]
num_shards = int(sys.argv[5])
prefixes = {
    "dover": ["dover_"],
    "pickscore": ["pickscore_"],
    "hpsv3": ["hpsv3_"],
}[metric]

for shard_idx in range(num_shards):
    path = output_root / f"{output_prefix}_{segment}_shard_{shard_idx:02d}" / "results.jsonl"
    if not path.exists():
        raise SystemExit(1)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(1)
    for row in rows:
        metrics = row.get("metrics", {})
        if not any(any(key.startswith(prefix) for prefix in prefixes) for key in metrics):
            raise SystemExit(1)
        if metric in row.get("errors", {}):
            raise SystemExit(1)
raise SystemExit(0)
PY
}

merge_segment() {
    local segment="$1"
    "$PYTHON" scripts/evaluation/merge_sharded_eval_results.py \
        --output-root "$OUTPUT_ROOT" \
        --output-name "${OUTPUT_PREFIX}_${segment}" \
        --shard-glob "${OUTPUT_PREFIX}_${segment}_shard_*"
}

run_vbench_shards() {
    local segment="$1"
    local model="$2"
    local port_base="$VBENCH_MASTER_PORT_BASE"
    VBENCH_MASTER_PORT_BASE=$((VBENCH_MASTER_PORT_BASE + NUM_SHARDS + 1))
    echo "[ratio-eval] vbench segment=$segment model=$model"
    mkdir -p "$OUTPUT_ROOT/logs/${OUTPUT_PREFIX}_${segment}_vbench"
    local pids=()
    for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
        local gpu="${GPU_LIST[$shard_idx]}"
        local shard_name
        shard_name="shard_$(printf '%02d' "$shard_idx")"
        local video_dir="$SHARD_ROOT/$segment/$shard_name/$model/videos"
        local prompt_json="$video_dir/vbench_prompts.json"
        local out_dir="$OUTPUT_ROOT/${OUTPUT_PREFIX}_${segment}_vbench/$model/$shard_name"
        (
            set -euo pipefail
            export CUDA_VISIBLE_DEVICES="$gpu"
            export MASTER_ADDR="127.0.0.1"
            export MASTER_PORT="$((port_base + shard_idx))"
            export WORLD_SIZE=1
            export RANK=0
            export LOCAL_RANK=0
            cmd=(
                "$PYTHON" "$VBENCH_ROOT/evaluate.py"
                --videos_path "$video_dir"
                --output_path "$out_dir"
                --mode custom_input
                --prompt_file "$prompt_json"
                --dimension $VBENCH_DIMS
            )
            if [ "$VBENCH_LOCAL" = "1" ] || [ "$VBENCH_LOCAL" = "true" ]; then
                cmd+=(--load_ckpt_from_local True)
            fi
            "${cmd[@]}"
        ) >"$OUTPUT_ROOT/logs/${OUTPUT_PREFIX}_${segment}_vbench/${model}_${shard_name}.log" 2>&1 &
        pids+=("$!")
        echo "[ratio-eval]   launched vbench $segment $model $shard_name on gpu $gpu pid=$!"
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "ERROR: one or more VBench shards failed for segment=$segment model=$model" >&2
        echo "Check logs in $OUTPUT_ROOT/logs/${OUTPUT_PREFIX}_${segment}_vbench" >&2
        exit 1
    fi
}

if [ "$RUN_QUALITY_METRICS" = "1" ]; then
    for segment in "${SEGMENT_LIST[@]}"; do
        for metric in "${METRIC_LIST[@]}"; do
            if [ "$SKIP_DONE_METRICS" = "1" ] && metric_done_for_segment "$segment" "$metric"; then
                echo "[ratio-eval] skip completed metric=$metric segment=$segment"
                continue
            fi
            run_metric_shards "$segment" "$metric"
        done
        merge_segment "$segment"
    done
fi

if [ "$RUN_VBENCH" = "1" ]; then
    for segment in "${SEGMENT_LIST[@]}"; do
        for model in "${MODEL_LIST[@]}"; do
            run_vbench_shards "$segment" "$model"
        done
    done
fi

"$PYTHON" scripts/evaluation/summarize_helios_long_ratio_eval.py \
    --output-root "$OUTPUT_ROOT" \
    --output-prefix "$OUTPUT_PREFIX" \
    --models "${MODEL_LIST[@]}" \
    --segments "${SEGMENT_LIST[@]}" \
    --metrics "${METRIC_LIST[@]}" \
    --vbench-dims $VBENCH_DIMS

echo "[ratio-eval] done"
echo "  summary: $OUTPUT_ROOT/${OUTPUT_PREFIX}_ratio_summary"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-python}
HELIOS_EVAL_DIR=${HELIOS_EVAL_DIR:-$PROJECT_ROOT/eval}
HELPER=${HELPER:-$PROJECT_ROOT/scripts/evaluation/heliosbench_on_our_prompts_helper.py}

DATASET_NAME=${DATASET_NAME:-long100}
SOURCE_ROOT=${SOURCE_ROOT:-tmp/outputs/helios_stage_long_100}
PROMPT_JSONL=${PROMPT_JSONL:-dataset/helios_single_prompts/prompts_with_categories.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-tmp/outputs/heliosbench-on-our-prompts/$DATASET_NAME}
DURATION_FRAMES=${DURATION_FRAMES:-231}

MODELS_STR=${MODELS:-base mid distilled}
GPUS_STR=${GPUS:-0 1 2 3 4 5 6 7}
NUM_SHARDS=${NUM_SHARDS:-8}
HEIGHT=${HEIGHT:-384}
WIDTH=${WIDTH:-640}
MOTION_NUM_WORKERS=${MOTION_NUM_WORKERS:-4}

METRICS_STR=${METRICS:-aesthetic motion_amplitude motion_smoothness semantic drifting_aesthetic drifting_motion_smoothness drifting_semantic}

CKPT_ROOT=${CKPT_ROOT:-$PROJECT_ROOT/tmp/models/eval_ckpts}
HELIOSBENCH_CKPT_ROOT=${HELIOSBENCH_CKPT_ROOT:-$CKPT_ROOT/HeliosBench}

if [[ -f "$HELIOSBENCH_CKPT_ROOT/aesthetic_model/ViT-L-14.pt" ]]; then
  CLIP_MODEL_PATH=${CLIP_MODEL_PATH:-$HELIOSBENCH_CKPT_ROOT/aesthetic_model/ViT-L-14.pt}
else
  CLIP_MODEL_PATH=${CLIP_MODEL_PATH:-$CKPT_ROOT/VBench/clip_model/ViT-L-14.pt}
fi

if [[ -f "$HELIOSBENCH_CKPT_ROOT/aesthetic_model/sa_0_4_vit_l_14_linear.pth" ]]; then
  AESTHETIC_MODEL_PATH=${AESTHETIC_MODEL_PATH:-$HELIOSBENCH_CKPT_ROOT/aesthetic_model/sa_0_4_vit_l_14_linear.pth}
else
  AESTHETIC_MODEL_PATH=${AESTHETIC_MODEL_PATH:-$CKPT_ROOT/VBench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth}
fi

if [[ -f "$HELIOSBENCH_CKPT_ROOT/amt_model/amt-s.pth" ]]; then
  SMOOTHNESS_MODEL_PATH=${SMOOTHNESS_MODEL_PATH:-$HELIOSBENCH_CKPT_ROOT/amt_model/amt-s.pth}
else
  SMOOTHNESS_MODEL_PATH=${SMOOTHNESS_MODEL_PATH:-$CKPT_ROOT/VBench/amt_model/amt-s.pth}
fi

if [[ -f "$HELIOSBENCH_CKPT_ROOT/AMT-S.yaml" ]]; then
  AMT_CONFIG=${AMT_CONFIG:-$HELIOSBENCH_CKPT_ROOT/AMT-S.yaml}
else
  AMT_CONFIG=${AMT_CONFIG:-$HELIOS_EVAL_DIR/utils/third_party/amt/cfgs/AMT-S_gopro.yaml}
fi
SEMANTIC_MODEL_PATH=${SEMANTIC_MODEL_PATH:-$HELIOSBENCH_CKPT_ROOT/ViCLIP}
SEMANTIC_TOKENIZER_COMPAT_PATH="$HELIOS_EVAL_DIR/checkpoints/ViCLIP/bpe_simple_vocab_16e6.txt.gz"

RESULTS_ROOT="$OUTPUT_ROOT/results"
LOG_ROOT="$OUTPUT_ROOT/logs"
TASKS_TSV="$OUTPUT_ROOT/tasks.tsv"
PROMPT_CSV="$OUTPUT_ROOT/prompts.csv"

IFS=' ' read -r -a MODELS_ARR <<< "$MODELS_STR"
IFS=' ' read -r -a GPUS_ARR <<< "$GPUS_STR"
IFS=' ' read -r -a METRICS_ARR <<< "$METRICS_STR"

if [[ ${#GPUS_ARR[@]} -ne "$NUM_SHARDS" ]]; then
  echo "ERROR: NUM_SHARDS=$NUM_SHARDS but GPUS has ${#GPUS_ARR[@]} entries: $GPUS_STR" >&2
  echo "Set NUM_SHARDS to match GPUS, or pass GPUS='0 1 ...'." >&2
  exit 1
fi

check_file() {
  local path=$1
  local label=$2
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing $label: $path" >&2
    exit 1
  fi
}

check_dir() {
  local path=$1
  local label=$2
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing $label: $path" >&2
    exit 1
  fi
}

check_file "$PYTHON" "python"
check_dir "$HELIOS_EVAL_DIR" "Helios eval dir"
check_file "$HELPER" "helper"
check_dir "$SOURCE_ROOT" "source video root"
check_file "$PROMPT_JSONL" "prompt jsonl"
check_file "$CLIP_MODEL_PATH" "CLIP ViT-L/14 checkpoint"
check_file "$AESTHETIC_MODEL_PATH" "LAION aesthetic checkpoint"
check_file "$SMOOTHNESS_MODEL_PATH" "AMT checkpoint"
check_file "$AMT_CONFIG" "AMT config"

if [[ "$METRICS_STR" == *semantic* ]]; then
  "$PYTHON" - <<'PY'
try:
    import pkg_resources  # noqa: F401
    from pkg_resources import packaging  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "ERROR: semantic metrics require pkg_resources from setuptools. "
        "Install a compatible version with: "
        "python -m pip install 'setuptools<81'"
    ) from exc
PY
  if [[ ! -f "$SEMANTIC_MODEL_PATH/ViClip-InternVid-10M-FLT.pth" || ! -f "$SEMANTIC_MODEL_PATH/bpe_simple_vocab_16e6.txt.gz" ]]; then
    cat >&2 <<EOF
ERROR: missing HeliosBench ViCLIP checkpoint under:
  $SEMANTIC_MODEL_PATH

Expected:
  $SEMANTIC_MODEL_PATH/ViClip-InternVid-10M-FLT.pth
  $SEMANTIC_MODEL_PATH/bpe_simple_vocab_16e6.txt.gz

Download it once with:
  HF_HOME=$CKPT_ROOT/.hf_cache huggingface-cli download BestWishYsh/HeliosBench-Weights --local-dir $HELIOSBENCH_CKPT_ROOT

Or rerun with SEMANTIC_MODEL_PATH=/path/to/ViCLIP.
EOF
    exit 1
  fi
  mkdir -p "$(dirname "$SEMANTIC_TOKENIZER_COMPAT_PATH")"
  if [[ ! -f "$SEMANTIC_TOKENIZER_COMPAT_PATH" ]]; then
    ln -s "$SEMANTIC_MODEL_PATH/bpe_simple_vocab_16e6.txt.gz" "$SEMANTIC_TOKENIZER_COMPAT_PATH"
  fi
  check_file "$SEMANTIC_TOKENIZER_COMPAT_PATH" "ViCLIP tokenizer compatibility path"
fi

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT"

echo "[heliosbench] project_root          : $PROJECT_ROOT"
echo "[heliosbench] dataset_name          : $DATASET_NAME"
echo "[heliosbench] source_root           : $SOURCE_ROOT"
echo "[heliosbench] prompt_jsonl          : $PROMPT_JSONL"
echo "[heliosbench] output_root           : $OUTPUT_ROOT"
echo "[heliosbench] models                : ${MODELS_ARR[*]}"
echo "[heliosbench] metrics               : ${METRICS_ARR[*]}"
echo "[heliosbench] gpus                  : ${GPUS_ARR[*]}"
echo "[heliosbench] python                : $PYTHON"
echo "[heliosbench] naturalness           : skipped"
echo "[heliosbench] throughput            : skipped"
echo "[heliosbench] semantic model path   : $SEMANTIC_MODEL_PATH"
echo "[heliosbench] semantic tokenizer    : $SEMANTIC_TOKENIZER_COMPAT_PATH"

"$PYTHON" "$HELPER" prepare \
  --source-root "$SOURCE_ROOT" \
  --prompt-jsonl "$PROMPT_JSONL" \
  --output-root "$OUTPUT_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --models "${MODELS_ARR[@]}" \
  --num-shards "$NUM_SHARDS" \
  --duration-frames "$DURATION_FRAMES"

run_one() {
  local metric=$1
  local task_name=$2
  local video_dir=$3
  local gpu=$4
  local log_path="$LOG_ROOT/${metric}_${task_name}.log"

  echo "[heliosbench] launch metric=$metric task=$task_name gpu=$gpu log=$log_path"
  (
    case "$metric" in
      aesthetic)
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HELPER" run-metric \
          --helios-eval-dir "$HELIOS_EVAL_DIR" \
          --metric "$metric" \
          --height "$HEIGHT" \
          --width "$WIDTH" \
          --input-csv "$PROMPT_CSV" \
          --video-dir "$video_dir" \
          --output-path "$RESULTS_ROOT" \
          --clip-model-path "$CLIP_MODEL_PATH" \
          --aesthetic-model-path "$AESTHETIC_MODEL_PATH"
        ;;
      motion_amplitude)
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HELPER" run-metric \
          --helios-eval-dir "$HELIOS_EVAL_DIR" \
          --metric "$metric" \
          --height "$HEIGHT" \
          --width "$WIDTH" \
          --input-csv "$PROMPT_CSV" \
          --video-dir "$video_dir" \
          --output-path "$RESULTS_ROOT" \
          --num-workers "$MOTION_NUM_WORKERS"
        ;;
      motion_smoothness)
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HELPER" run-metric \
          --helios-eval-dir "$HELIOS_EVAL_DIR" \
          --metric "$metric" \
          --height "$HEIGHT" \
          --width "$WIDTH" \
          --input-csv "$PROMPT_CSV" \
          --video-dir "$video_dir" \
          --output-path "$RESULTS_ROOT" \
          --amt-config "$AMT_CONFIG" \
          --smoothness-model-path "$SMOOTHNESS_MODEL_PATH"
        ;;
      semantic)
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HELPER" run-metric \
          --helios-eval-dir "$HELIOS_EVAL_DIR" \
          --metric "$metric" \
          --height "$HEIGHT" \
          --width "$WIDTH" \
          --input-csv "$PROMPT_CSV" \
          --video-dir "$video_dir" \
          --output-path "$RESULTS_ROOT" \
          --semantic-model-path "$SEMANTIC_MODEL_PATH"
        ;;
      drifting_aesthetic)
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HELPER" run-metric \
          --helios-eval-dir "$HELIOS_EVAL_DIR" \
          --metric "$metric" \
          --height "$HEIGHT" \
          --width "$WIDTH" \
          --input-csv "$PROMPT_CSV" \
          --video-dir "$video_dir" \
          --output-path "$RESULTS_ROOT" \
          --clip-model-path "$CLIP_MODEL_PATH" \
          --aesthetic-model-path "$AESTHETIC_MODEL_PATH"
        ;;
      drifting_motion_smoothness)
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HELPER" run-metric \
          --helios-eval-dir "$HELIOS_EVAL_DIR" \
          --metric "$metric" \
          --height "$HEIGHT" \
          --width "$WIDTH" \
          --input-csv "$PROMPT_CSV" \
          --video-dir "$video_dir" \
          --output-path "$RESULTS_ROOT" \
          --amt-config "$AMT_CONFIG" \
          --smoothness-model-path "$SMOOTHNESS_MODEL_PATH"
        ;;
      drifting_semantic)
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HELPER" run-metric \
          --helios-eval-dir "$HELIOS_EVAL_DIR" \
          --metric "$metric" \
          --height "$HEIGHT" \
          --width "$WIDTH" \
          --input-csv "$PROMPT_CSV" \
          --video-dir "$video_dir" \
          --output-path "$RESULTS_ROOT" \
          --semantic-model-path "$SEMANTIC_MODEL_PATH"
        ;;
      *)
        echo "Unknown metric: $metric" >&2
        exit 2
        ;;
    esac
  ) >"$log_path" 2>&1
}

run_metric() {
  local metric=$1
  echo "[heliosbench] metric=$metric start"
  local batch_pids=()
  local batch_names=()
  local task_idx=0
  local failed=0

  while IFS=$'\t' read -r task_name model shard video_dir prompt_csv num_videos; do
    if [[ "$task_name" == "task_name" ]]; then
      continue
    fi
    local gpu=${GPUS_ARR[$((task_idx % ${#GPUS_ARR[@]}))]}
    run_one "$metric" "$task_name" "$video_dir" "$gpu" &
    batch_pids+=("$!")
    batch_names+=("$task_name")
    task_idx=$((task_idx + 1))

    if [[ ${#batch_pids[@]} -eq ${#GPUS_ARR[@]} ]]; then
      for i in "${!batch_pids[@]}"; do
        if ! wait "${batch_pids[$i]}"; then
          echo "ERROR: metric=$metric task=${batch_names[$i]} failed; see $LOG_ROOT/${metric}_${batch_names[$i]}.log" >&2
          failed=1
        fi
      done
      batch_pids=()
      batch_names=()
      if [[ "$failed" -ne 0 ]]; then
        exit 1
      fi
    fi
  done < "$TASKS_TSV"

  for i in "${!batch_pids[@]}"; do
    if ! wait "${batch_pids[$i]}"; then
      echo "ERROR: metric=$metric task=${batch_names[$i]} failed; see $LOG_ROOT/${metric}_${batch_names[$i]}.log" >&2
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    exit 1
  fi
  echo "[heliosbench] metric=$metric done"
}

for metric in "${METRICS_ARR[@]}"; do
  run_metric "$metric"
done

"$PYTHON" "$HELPER" merge \
  --output-root "$OUTPUT_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --models "${MODELS_ARR[@]}" \
  --num-shards "$NUM_SHARDS" \
  --metrics "${METRICS_ARR[@]}" \
  --helios-merge-script "$HELIOS_EVAL_DIR/9_merge_all_scores.py"

echo "[heliosbench] done"
echo "[heliosbench] merged summary: $OUTPUT_ROOT/results_merged_no_naturalness/summary_no_naturalness.csv"

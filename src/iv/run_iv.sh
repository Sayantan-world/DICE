#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../scripts/common.sh"

INPUT_JSONL="${INPUT_JSONL:-$(ensure_policy_data)/all.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-outputs/iv/target_observed_iv.jsonl}"
STATS_JSON="${STATS_JSON:-outputs/iv/target_observed_iv_stats.json}"
ERRORS_JSONL="${ERRORS_JSONL:-outputs/iv/target_observed_iv_errors.jsonl}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
STUDENT_ADAPTER_PATH="${STUDENT_ADAPTER_PATH:-}"
JUDGE_MODEL_NAMES="${JUDGE_MODEL_NAMES:-moonshotai/kimi-k2-instruct-0905,openai/gpt-oss-120b,meta/llama-3.3-70b-instruct}"

FILTER_VARIANT="${FILTER_VARIANT:-target}"
FILTER_PROVENANCE="${FILTER_PROVENANCE:-observed_multi_dialogue}"
K_ROLLOUTS="${K_ROLLOUTS:-4}"
LIMIT="${LIMIT:-}"

mkdir -p outputs/iv

ADAPTER_FLAGS=()
if [[ -n "${STUDENT_ADAPTER_PATH}" ]]; then
  ADAPTER_FLAGS+=(--student_adapter_path "${STUDENT_ADAPTER_PATH}")
fi
LIMIT_FLAGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_FLAGS+=(--limit "${LIMIT}")
fi

python src/iv/compute_iv.py \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --stats_json "${STATS_JSON}" \
  --errors_jsonl "${ERRORS_JSONL}" \
  --student_model_name "${STUDENT_MODEL_NAME}" \
  --judge_model_names "${JUDGE_MODEL_NAMES}" \
  --filter_variant "${FILTER_VARIANT}" \
  --filter_provenance "${FILTER_PROVENANCE}" \
  --k_rollouts "${K_ROLLOUTS}" \
  --seed 42 \
  --bf16 \
  --use_4bit \
  --min_api_interval_seconds 2.0 \
  "${ADAPTER_FLAGS[@]}" \
  "${LIMIT_FLAGS[@]}"

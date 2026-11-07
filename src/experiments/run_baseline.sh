#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../scripts/common.sh"

# Direct QA baselines: zeroshot / CoT / gold-step CoT.

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
INPUT_JSON="${INPUT_JSON:-$(ensure_eval_json triplets_250)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/preds/qa_baseline}"
RESULT_ROOT="${RESULT_ROOT:-outputs/metrics/qa_baseline}"
MODE="${MODE:-zeroshot,cot,steps_augmented_cot}"
VARIANTS="${VARIANTS:-target,easy,hard}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-0}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"

VARIANT_FLAGS=()
IFS=',' read -r -a _variant_arr <<< "${VARIANTS}"
for v in "${_variant_arr[@]}"; do
  case "$(echo "${v}" | xargs)" in
    target) VARIANT_FLAGS+=(--target) ;;
    easy) VARIANT_FLAGS+=(--easy) ;;
    hard) VARIANT_FLAGS+=(--hard) ;;
  esac
done

IFS=',' read -r -a _mode_arr <<< "${MODE}"
for m in "${_mode_arr[@]}"; do
  m="$(echo "${m}" | xargs)"
  MODE_FLAG="--zeroshot"
  if [[ "${m}" == "cot" ]]; then
    MODE_FLAG="--cot"
  elif [[ "${m}" == "steps_augmented_cot" ]]; then
    MODE_FLAG="--steps_augmented_cot"
  elif [[ "${m}" != "zeroshot" ]]; then
    echo "Skipping unsupported mode: ${m}"
    continue
  fi
  echo "[RUN] QA baseline mode=${m} model=${MODEL_NAME}"
  python src/experiments/baseline.py \
    --model_name "${MODEL_NAME}" \
    --input_json "${INPUT_JSON}" \
    --output_root "${OUTPUT_ROOT}" \
    --result_root "${RESULT_ROOT}" \
    --max_new_tokens 256 \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --pipeline_parallel_size "${PIPELINE_PARALLEL_SIZE}" \
    --gpu_memory_utilization 0.92 \
    --max_num_seqs 256 \
    --max_model_len 4096 \
    --dtype auto \
    "${MODE_FLAG}" \
    "${VARIANT_FLAGS[@]}"
done

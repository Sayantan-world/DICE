#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../scripts/common.sh"

INPUT_JSON="${INPUT_JSON:-$(ensure_eval_json triplets_250)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/preds/socraticlm_single}"
RESULT_ROOT="${RESULT_ROOT:-outputs/metrics/socraticlm_single}"
VARIANTS="${VARIANTS:-target,easy,hard}"
MAX_TURNS="${MAX_TURNS:-8}"

TEACHER_TENSOR_PARALLEL_SIZE="${TEACHER_TENSOR_PARALLEL_SIZE:-1}"
TEACHER_PIPELINE_PARALLEL_SIZE="${TEACHER_PIPELINE_PARALLEL_SIZE:-3}"
STUDENT_DEVICE="${STUDENT_DEVICE:-3}"

VARIANT_FLAGS=()
IFS=',' read -r -a _variant_arr <<< "${VARIANTS}"
for v in "${_variant_arr[@]}"; do
  case "$(echo "${v}" | xargs)" in
    target) VARIANT_FLAGS+=(--target) ;;
    easy) VARIANT_FLAGS+=(--easy) ;;
    hard) VARIANT_FLAGS+=(--hard) ;;
  esac
done

python src/experiments/socraticlm_single.py \
  --input_json "${INPUT_JSON}" \
  --output_root "${OUTPUT_ROOT}" \
  --result_root "${RESULT_ROOT}" \
  --max_turns "${MAX_TURNS}" \
  --teacher_model "${TEACHER_MODEL:-CogBase-USTC/SocraticLM}" \
  --student_model "${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --teacher_max_new_tokens 256 \
  --student_max_new_tokens 256 \
  --teacher_temperature 0.2 \
  --student_temperature 0.6 \
  --top_p 0.95 \
  --teacher_tensor_parallel_size "${TEACHER_TENSOR_PARALLEL_SIZE}" \
  --teacher_pipeline_parallel_size "${TEACHER_PIPELINE_PARALLEL_SIZE}" \
  --student_device "${STUDENT_DEVICE}" \
  --teacher_gpu_memory_utilization 0.90 \
  --teacher_max_num_seqs 1 \
  --max_model_len 2048 \
  --dtype auto \
  "${VARIANT_FLAGS[@]}"

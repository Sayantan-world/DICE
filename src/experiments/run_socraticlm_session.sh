#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../scripts/common.sh"

INPUT_JSON="${INPUT_JSON:-$(ensure_eval_json triplets_250)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/preds/socraticlm_session}"
RESULT_ROOT="${RESULT_ROOT:-outputs/metrics/socraticlm_session}"
LIMIT="${LIMIT:-}"

TEACHER_TENSOR_PARALLEL_SIZE="${TEACHER_TENSOR_PARALLEL_SIZE:-1}"
TEACHER_PIPELINE_PARALLEL_SIZE="${TEACHER_PIPELINE_PARALLEL_SIZE:-3}"
STUDENT_DEVICE="${STUDENT_DEVICE:-3}"

LIMIT_FLAGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_FLAGS+=(--limit "${LIMIT}")
fi

python src/experiments/dice_session_sim.py \
  --input_json "${INPUT_JSON}" \
  --output_root "${OUTPUT_ROOT}" \
  --result_root "${RESULT_ROOT}" \
  --max_session_turns 20 \
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
  --max_model_len 4096 \
  --dtype auto \
  --response_only \
  "${LIMIT_FLAGS[@]}"

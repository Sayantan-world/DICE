#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../scripts/common.sh"

# Shared-budget session: easy -> target -> hard, 20 turns.

INPUT_JSON="${INPUT_JSON:-$(ensure_eval_json triplets_250)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/preds/dice_session}"
RESULT_ROOT="${RESULT_ROOT:-outputs/metrics/dice_session}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
POLICY_BASE_MODEL="${POLICY_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
POLICY_ADAPTER_PATH="${POLICY_ADAPTER_PATH:-chocokiddo/DICE-Policy}"

DICE_MODE="${DICE_MODE:-dice_cost}"
INTERVENTION_COST_LAMBDA="${INTERVENTION_COST_LAMBDA:-0.25}"
INTERVENTION_THRESHOLD_TAU="${INTERVENTION_THRESHOLD_TAU:-0.8}"
MAX_SESSION_TURNS="${MAX_SESSION_TURNS:-20}"
LIMIT="${LIMIT:-}"

USE_VERIFIER="${USE_VERIFIER:-1}"
ACCEPTANCE_OVERRIDE_ACTION="${ACCEPTANCE_OVERRIDE_ACTION:-ask_for_justification}"

TEACHER_TENSOR_PARALLEL_SIZE="${TEACHER_TENSOR_PARALLEL_SIZE:-1}"
TEACHER_PIPELINE_PARALLEL_SIZE="${TEACHER_PIPELINE_PARALLEL_SIZE:-3}"
STUDENT_DEVICE="${STUDENT_DEVICE:-3}"
POLICY_DEVICE="${POLICY_DEVICE:-3}"

if [[ -z "${POLICY_ADAPTER_PATH}" ]]; then
  echo "Set POLICY_ADAPTER_PATH to the released DICE policy (Hugging Face id or local dir)."
  exit 1
fi

MODE_FLAGS=()
case "${DICE_MODE}" in
  dice_map) MODE_FLAGS+=(--dice_map) ;;
  dice_cost) MODE_FLAGS+=(--dice_cost --intervention_cost_lambda "${INTERVENTION_COST_LAMBDA}") ;;
  dice_threshold) MODE_FLAGS+=(--dice_threshold --intervention_threshold_tau "${INTERVENTION_THRESHOLD_TAU}") ;;
  *) echo "DICE_MODE must be dice_map, dice_cost, or dice_threshold"; exit 1 ;;
esac

VERIFIER_FLAGS=(--acceptance_override_action "${ACCEPTANCE_OVERRIDE_ACTION}")
if [[ "${USE_VERIFIER}" == "1" ]]; then
  VERIFIER_FLAGS+=(--use_acceptance_verifier)
else
  VERIFIER_FLAGS+=(--no-use_acceptance_verifier)
fi

LIMIT_FLAGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_FLAGS+=(--limit "${LIMIT}")
fi

python src/experiments/dice_session_sim.py \
  --input_json "${INPUT_JSON}" \
  --output_root "${OUTPUT_ROOT}" \
  --result_root "${RESULT_ROOT}" \
  --max_session_turns "${MAX_SESSION_TURNS}" \
  --teacher_model "${TEACHER_MODEL}" \
  --student_model "${STUDENT_MODEL}" \
  --policy_base_model "${POLICY_BASE_MODEL}" \
  --policy_adapter_path "${POLICY_ADAPTER_PATH}" \
  --teacher_max_new_tokens 256 \
  --student_max_new_tokens 256 \
  --teacher_temperature 0.2 \
  --student_temperature 0.6 \
  --top_p 0.95 \
  --teacher_tensor_parallel_size "${TEACHER_TENSOR_PARALLEL_SIZE}" \
  --teacher_pipeline_parallel_size "${TEACHER_PIPELINE_PARALLEL_SIZE}" \
  --student_device "${STUDENT_DEVICE}" \
  --policy_device "${POLICY_DEVICE}" \
  --teacher_gpu_memory_utilization 0.90 \
  --teacher_max_num_seqs 1 \
  --max_model_len 4096 \
  --dtype auto \
  --policy_score_normalization mean \
  --policy_use_4bit \
  --policy_bf16 \
  "${MODE_FLAGS[@]}" \
  "${VERIFIER_FLAGS[@]}" \
  "${LIMIT_FLAGS[@]}"

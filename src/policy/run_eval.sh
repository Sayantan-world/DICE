#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../scripts/common.sh"

TEST_JSONL="${TEST_JSONL:-$(ensure_policy_data)/test.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-${POLICY_ADAPTER_PATH:-chocokiddo/DICE-Policy}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/policy/eval}"

if [[ -z "${ADAPTER_PATH}" ]]; then
  echo "Set ADAPTER_PATH or POLICY_ADAPTER_PATH to the DICE policy (Hugging Face id or local dir)."
  exit 1
fi

python src/policy/evaluate.py \
  --test_jsonl "${TEST_JSONL}" \
  --model_name "${MODEL_NAME}" \
  --adapter_path "${ADAPTER_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length 4096 \
  --bf16 \
  --use_4bit \
  --action_only

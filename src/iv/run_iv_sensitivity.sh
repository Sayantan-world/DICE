#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../scripts/common.sh"

IV_JSONL="${IV_JSONL:-$(ensure_iv_data)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/iv/sensitivity}"
DATASET_NAME="${DATASET_NAME:-target_observed}"

python src/iv/analyze_iv_sensitivity.py \
  --output_dir "${OUTPUT_DIR}" \
  --epsilons "0,0.05,0.10,0.125,0.25,0.375,0.50" \
  --input "${DATASET_NAME}=${IV_JSONL}"

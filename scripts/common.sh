#!/usr/bin/env bash
# Source from any repo script:  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/common.sh"
# when the caller lives in src/<module>/.

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${COMMON_DIR}/.." && pwd)"
export ROOT_DIR

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

export PYTHONPATH="${ROOT_DIR}/src/experiments:${ROOT_DIR}/src/iv:${ROOT_DIR}/src/policy:${ROOT_DIR}/third_party/stratl${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"

mkdir -p "${ROOT_DIR}/outputs/preds" "${ROOT_DIR}/outputs/metrics" "${ROOT_DIR}/outputs/iv" "${ROOT_DIR}/outputs/policy" "${ROOT_DIR}/data"

# Download DICE-Bench JSON if missing. Prints the local path on stdout.
ensure_eval_json() {
  local split="${1:-triplets_250}"
  local dest="${ROOT_DIR}/data/${split}.json"
  if [[ ! -f "${dest}" ]]; then
    python "${ROOT_DIR}/scripts/download_data.py" \
      --split "${split}" \
      --output_dir "${ROOT_DIR}/data"
  fi
  printf '%s\n' "${dest}"
}

ensure_policy_data() {
  if [[ ! -f "${ROOT_DIR}/data/policy/train.jsonl" ]]; then
    python "${ROOT_DIR}/scripts/download_data.py" \
      --split policy \
      --output_dir "${ROOT_DIR}/data"
  fi
  printf '%s\n' "${ROOT_DIR}/data/policy"
}

ensure_iv_data() {
  if [[ ! -f "${ROOT_DIR}/data/iv/target_observed.jsonl" ]]; then
    python "${ROOT_DIR}/scripts/download_data.py" \
      --split iv \
      --output_dir "${ROOT_DIR}/data"
  fi
  printf '%s\n' "${ROOT_DIR}/data/iv/target_observed.jsonl"
}

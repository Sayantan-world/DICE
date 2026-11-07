#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../scripts/common.sh"

SPLIT_DIR="${SPLIT_DIR:-$(ensure_policy_data)}"
INPUT_JSONL="${INPUT_JSONL:-${SPLIT_DIR}/all.jsonl}"
IV_JSONL="${IV_JSONL:-$(ensure_iv_data)}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
SKIP_KL="${SKIP_KL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/policy/qwen25_7b_tutor_policy_ivkl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "Missing ${INPUT_JSONL}. Download annotations with: python scripts/download_data.py --split policy"
  exit 1
fi

# Paper Eq. (2) always includes KL toward π_SFT (β = 0.05).
KL_FLAGS=()
if [[ "${SKIP_KL}" == "1" ]]; then
  echo "Warning: SKIP_KL=1 — training without KL. This is not the paper objective."
elif [[ -z "${SFT_ADAPTER}" ]]; then
  echo "Paper training uses IV-weighted CE + KL to an SFT policy (Eq. 2, β=0.05)."
  echo "Set SFT_ADAPTER to the Stage-1 SFT LoRA path, for example:"
  echo "  SFT_ADAPTER=/path/to/qwen25_7b_tutor_policy_lora bash src/policy/run_policy.sh"
  echo "To train IV-weighted SFT only (not the paper recipe): SKIP_KL=1 bash src/policy/run_policy.sh"
  exit 1
else
  KL_FLAGS+=(
    --use_kl_regularization
    --kl_ref_adapter_path "${SFT_ADAPTER}"
    --kl_beta 0.05
  )
fi

IV_FLAGS=()
if [[ -f "${IV_JSONL}" ]]; then
  IV_FLAGS+=(
    --use_iv_weighting
    --iv_jsonl "${IV_JSONL}"
    --iv_apply_variant target
    --iv_apply_provenance observed_multi_dialogue
    --missing_iv_weight 1.0
    --accept_weight 1.0
    --min_iv_weight 0.0
    --max_iv_weight 1.0
  )
else
  echo "Warning: ${IV_JSONL} not found; training without IV weights."
fi

python src/policy/train.py \
  --input_jsonl "${INPUT_JSONL}" \
  --split_dir "${SPLIT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --seed 42 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --early_stopping_patience 2 \
  --report_to "${REPORT_TO:-wandb}" \
  --wandb_project "${WANDB_PROJECT:-DICE}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --run_name "${RUN_NAME:-qwen25_7b_action_only_ivkl}" \
  --bf16 \
  --gradient_checkpointing \
  --use_4bit \
  --action_only \
  "${IV_FLAGS[@]}" \
  "${KL_FLAGS[@]}"

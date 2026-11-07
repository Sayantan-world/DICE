# DICE

**DICE: Decoupling Capability from Intervention Necessity in LLM Tutoring**

This repository contains the code used for the paper experiments: intervention-value (IV) estimation, policy training, and tutoring evaluation. DICE-Bench data is released on Hugging Face; this repo does not include the dataset generation pipeline.

The paper policy is the merged model [chocokiddo/DICE-Policy](https://huggingface.co/chocokiddo/DICE-Policy). Experiment runners use it by default.

## Install

```bash
git clone https://github.com/Sayantan-world/DICE.git
cd DICE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py --split triplets
```

Copy `.env.example` to `.env` and fill in tokens as needed:

```bash
cp .env.example .env
```

- `HF_TOKEN` — gated Hugging Face models (Llama)
- `NVIDIA_API_KEY` — optional IV judges (only if you recompute IV)
- `OPENAI_API_KEY` — optional OpenAI-compatible judges
- `WANDB_API_KEY` — policy training logs

Scripts auto-load `.env` from the repo root.

### Hardware

Paper runs used **4× A5000**, or a **single A100 / H100**. The 4-GPU setup serves the teacher with vLLM (`TP=1`, `PP=3`) on GPUs `0–2`; the student and policy sit on GPU `3`.

Override if you have fewer GPUs, for example 2 GPUs:

```bash
export TEACHER_PIPELINE_PARALLEL_SIZE=1
export STUDENT_DEVICE=1
export POLICY_DEVICE=1
```

## Data

Evaluation triplets are on Hugging Face: [chocokiddo/DICE-Bench](https://huggingface.co/datasets/chocokiddo/DICE-Bench).

| Split | Size | Use |
|---|---|---|
| `triplets_250` | 250 | Primary evaluation split for reported tutoring tables |
| `triplets_all` | 10,273 | Full easy / target / hard corpus |

```bash
python scripts/download_data.py --split triplets   # eval JSON
python scripts/download_data.py --split policy     # 20,477 action labels
python scripts/download_data.py --split iv         # Intervention Value estimates
python scripts/download_data.py --split all        # everything
```

Eval runners download `triplets_250` automatically. Policy training downloads the `policy` and `iv` configs if they are missing.

```python
from datasets import load_dataset
ds = load_dataset("chocokiddo/DICE-Bench", split="triplets_250")
policy = load_dataset("chocokiddo/DICE-Bench", "policy")
iv = load_dataset("chocokiddo/DICE-Bench", "iv", split="target_observed")
```

## Models

| Role | Default |
|---|---|
| Teacher | `Qwen/Qwen2.5-7B-Instruct` |
| Student (main tables) | `Qwen/Qwen2.5-1.5B-Instruct` |
| Student (Llama transfer) | `meta-llama/Llama-3.2-1B-Instruct` |
| SocraticLM baseline | `CogBase-USTC/SocraticLM` |
| DICE policy | [`chocokiddo/DICE-Policy`](https://huggingface.co/chocokiddo/DICE-Policy) |

The policy is a **merged** `Qwen2.5-7B-Instruct` checkpoint (IV-weighted action-only LoRA + KL, then merged). Runners load it as a full model; a PEFT adapter path still works if you train your own.

## Reproduce paper experiments

Run everything from the repo root. Defaults download and read `data/triplets_250.json`.

### 1. SocraticLM tutoring

Per-variant (8 turns):

```bash
bash src/experiments/run_socraticlm_single.sh
```

Session (easy → target → hard, 20-turn budget):

```bash
bash src/experiments/run_socraticlm_session.sh
```

### 2. DICE tutoring

The runners default `POLICY_ADAPTER_PATH` to `chocokiddo/DICE-Policy`. Override only if you have a local checkpoint.

**DICE-Cost** (paper default, λ = 0.25), per-variant:

```bash
DICE_MODE=dice_cost INTERVENTION_COST_LAMBDA=0.25 \
  bash src/experiments/run_dice_single.sh
```

Session:

```bash
DICE_MODE=dice_cost INTERVENTION_COST_LAMBDA=0.25 \
  bash src/experiments/run_dice_session.sh
```

**DICE-MAP:**

```bash
DICE_MODE=dice_map bash src/experiments/run_dice_single.sh
DICE_MODE=dice_map bash src/experiments/run_dice_session.sh
```

**DICE-Threshold** (example τ = 0.8):

```bash
DICE_MODE=dice_threshold INTERVENTION_THRESHOLD_TAU=0.8 \
  bash src/experiments/run_dice_single.sh
```

Smoke-test a few triplets:

```bash
LIMIT=2 DICE_MODE=dice_map bash src/experiments/run_dice_session.sh
```

Llama student (same DICE session protocol):

```bash
STUDENT_MODEL=meta-llama/Llama-3.2-1B-Instruct \
  DICE_MODE=dice_map \
  bash src/experiments/run_dice_session.sh
```

Outputs go to `outputs/preds/` and `outputs/metrics/`.

## Optional: recompute IV or retrain the policy

These steps are **not** required to reproduce the tutoring tables if you use [chocokiddo/DICE-Policy](https://huggingface.co/chocokiddo/DICE-Policy). Annotations and IV estimates are on Hugging Face.

### A. Intervention values

Needs `NVIDIA_API_KEY` only if you recompute IV from the released policy labels:

```bash
bash src/iv/run_iv.sh
```

Offline audit (HELPFUL / NEUTRAL / HARMFUL vs ε and score maps):

```bash
bash src/iv/run_iv_sensitivity.sh
```

### B. Train the policy

Uses the released `policy` + `iv` configs (downloaded automatically). The paper objective (Eq. 2) is IV-weighted action-only CE, sample-count normalized, plus KL to a Stage-1 SFT policy with β = 0.05.

Train the SFT reference first (same data, **no** `--use_iv_weighting` / `--use_kl_regularization`), then:

```bash
SFT_ADAPTER=/path/to/qwen25_7b_tutor_policy_lora \
  bash src/policy/run_policy.sh
```

`SFT_ADAPTER` is required. `SKIP_KL=1` trains IV-weighted SFT only and is **not** the paper recipe. Tutoring tables do not need a retrain; they use [chocokiddo/DICE-Policy](https://huggingface.co/chocokiddo/DICE-Policy).

Evaluate action accuracy on the held-out test split:

```bash
bash src/policy/run_eval.sh
```

## Repository layout

```
scripts/download_data.py      DICE-Bench download (Hugging Face)
src/iv/                       intervention-value estimation and sensitivity
src/policy/                   LoRA training and action-head eval
src/experiments/              SocraticLM and DICE tutoring runners
scripts/common.sh             repo-root + PYTHONPATH + .env
```

## Citation

```bibtex
@misc{pal2026dicedecouplingcapabilityintervention,
      title={DICE: Decoupling Capability from Intervention Necessity in LLM Tutoring}, 
      author={Sayantan Pal and Kaiyi Ji and Rohini K. Srihari},
      year={2026},
      eprint={2610.04825},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.04825}, 
}
```

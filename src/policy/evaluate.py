from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from tqdm import tqdm


# =========================================================
# V1 schema (must match training)
# =========================================================

LOCAL_LABELS = ["correct", "incorrect", "partial"]
GLOBAL_LABELS = ["on_track", "ahead", "off_track", "stalled"]
ISSUE_LABELS = [
    "none",
    "arithmetic_error",
    "student_question",
    "subgoal_skip",
    "conceptual_issue",
    "weak_justification",
]
ACTION_LABELS = [
    "accept_and_advance",
    "arithmetic_repair",
    "handle_question",
    "refocus_subgoal",
    "conceptual_hint",
    "ask_for_justification",
]

SYSTEM_PROMPT = """You are a tutoring-policy model.
Given a math tutoring context, diagnose the student's current reasoning state and choose the best next pedagogical action.
""".strip()

ACTION_ONLY_SYSTEM_PROMPT = """You are a tutoring-policy model.
Given a math tutoring context, choose the best next pedagogical action.
""".strip()


# =========================================================
# IO helpers
# =========================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{i}") from e
    return rows


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================
# Formatting (must match training)
# =========================================================

def normalize_history(history: Any) -> str:
    if not history:
        return "None"
    out = []
    for turn in history:
        if isinstance(turn, list) and len(turn) == 2:
            out.append(f"Teacher: {turn[0]}\nStudent: {turn[1]}")
    return "\n".join(out) if out else "None"


def format_user_prompt(row: Dict[str, Any]) -> str:
    question = row.get("question", "")
    analysis = row.get("analysis", "")
    answer = row.get("answer", "")
    current_subgoal = row.get("current_subgoal", "")
    teacher_question = row.get("teacher_question", "")
    student_response = row.get("student_response", "")
    history = normalize_history(row.get("history"))

    return f"""[PROBLEM]
{question}

[REFERENCE_ANALYSIS]
{analysis}

[REFERENCE_ANSWER]
{answer}

[CURRENT_SUBGOAL]
{current_subgoal}

[TEACHER_QUESTION]
{teacher_question}

[HISTORY]
{history}

[STUDENT_RESPONSE]
{student_response}"""


def build_prompt_messages(row: Dict[str, Any], action_only: bool = False) -> List[Dict[str, str]]:
    system_prompt = ACTION_ONLY_SYSTEM_PROMPT if action_only else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_user_prompt(row)},
    ]


# =========================================================
# Model loading
# =========================================================

def is_peft_adapter(path: str) -> bool:
    local = Path(path)
    if local.exists():
        return (local / "adapter_config.json").is_file()
    try:
        from huggingface_hub import file_exists

        return bool(file_exists(repo_id=path, filename="adapter_config.json"))
    except Exception:
        return False


def load_model_and_tokenizer(
    model_name: str,
    adapter_path: Optional[str] = None,
    use_4bit: bool = False,
    bf16: bool = False,
):
    load_as_adapter = bool(adapter_path) and is_peft_adapter(adapter_path)
    model_src = model_name if (not adapter_path or load_as_adapter) else adapter_path

    tokenizer = AutoTokenizer.from_pretrained(model_src, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_src,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    if load_as_adapter:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# =========================================================
# Constrained scoring
# =========================================================

def render_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


@torch.no_grad()
def score_candidate_text(
    model,
    tokenizer,
    prompt_text: str,
    candidate_text: str,
    max_length: int = 4096,
) -> float:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + candidate_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        prompt_ids = prompt_ids[-max_length:]

    input_ids = torch.tensor([full_ids], device=model.device)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]

    log_probs = torch.log_softmax(logits, dim=-1)

    candidate_token_start = min(len(prompt_ids), len(full_ids) - 1)
    total_lp = 0.0
    count = 0

    for pos in range(candidate_token_start, len(full_ids)):
        if pos == 0:
            continue
        pred_index = pos - 1
        tok_id = full_ids[pos]
        total_lp += log_probs[0, pred_index, tok_id].item()
        count += 1

    if count == 0:
        return -1e9
    return total_lp


def choose_best_candidate(
    model,
    tokenizer,
    prompt_text: str,
    candidates: List[str],
    max_length: int = 4096,
) -> Tuple[str, Dict[str, float]]:
    scores = {}
    for cand in candidates:
        scores[cand] = score_candidate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            candidate_text=cand,
            max_length=max_length,
        )
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best, scores


def predict_action_only(
    model,
    tokenizer,
    row: Dict[str, Any],
    max_length: int = 4096,
) -> Dict[str, Any]:
    messages = build_prompt_messages(row, action_only=True)
    prompt = render_prompt(tokenizer, messages)

    prefix = prompt + "[ACTION]\n"
    action_pred, action_scores = choose_best_candidate(
        model, tokenizer, prefix, ACTION_LABELS, max_length
    )

    structured_text = "[ACTION]\n" + action_pred

    return {
        "pred_best_action": action_pred,
        "score_breakdown": {
            "action": action_scores,
        },
        "structured_prediction": structured_text,
    }


def predict_structured_labels(
    model,
    tokenizer,
    row: Dict[str, Any],
    max_length: int = 4096,
) -> Dict[str, Any]:
    messages = build_prompt_messages(row, action_only=False)
    prompt = render_prompt(tokenizer, messages)

    prefix = prompt + "[DIAGNOSIS]\nlocal: "
    local_pred, local_scores = choose_best_candidate(model, tokenizer, prefix, LOCAL_LABELS, max_length)

    prefix = prompt + f"[DIAGNOSIS]\nlocal: {local_pred}\nglobal: "
    global_pred, global_scores = choose_best_candidate(model, tokenizer, prefix, GLOBAL_LABELS, max_length)

    prefix = (
        prompt
        + f"[DIAGNOSIS]\nlocal: {local_pred}\n"
        + f"global: {global_pred}\n"
        + "issue: "
    )
    issue_pred, issue_scores = choose_best_candidate(model, tokenizer, prefix, ISSUE_LABELS, max_length)

    prefix = (
        prompt
        + f"[DIAGNOSIS]\nlocal: {local_pred}\n"
        + f"global: {global_pred}\n"
        + f"issue: {issue_pred}\n"
        + "[ACTION]\n"
    )
    action_pred, action_scores = choose_best_candidate(model, tokenizer, prefix, ACTION_LABELS, max_length)

    structured_text = (
        "[DIAGNOSIS]\n"
        f"local: {local_pred}\n"
        f"global: {global_pred}\n"
        f"issue: {issue_pred}\n"
        "[ACTION]\n"
        f"{action_pred}"
    )

    return {
        "pred_local_correctness": local_pred,
        "pred_global_trajectory": global_pred,
        "pred_reasoning_issue": issue_pred,
        "pred_best_action": action_pred,
        "score_breakdown": {
            "local": local_scores,
            "global": global_scores,
            "issue": issue_scores,
            "action": action_scores,
        },
        "structured_prediction": structured_text,
    }


# =========================================================
# Evaluation
# =========================================================

def compute_head_metrics(golds: List[str], preds: List[str], labels: List[str]) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(golds, preds)),
        "macro_f1": float(f1_score(golds, preds, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(golds, preds, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            golds, preds, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(golds, preds, labels=labels).tolist(),
        "labels": labels,
    }


def eval_subset(rows: List[Dict[str, Any]], action_only: bool = False) -> Dict[str, Any]:
    if not rows:
        return {"num_rows": 0}

    gold_action = [r["gold"]["best_action"] for r in rows]
    pred_action = [r["pred"]["pred_best_action"] for r in rows]

    over_intervention = 0
    under_intervention = 0
    for g, p in zip(gold_action, pred_action):
        gold_intervene = g != "accept_and_advance"
        pred_intervene = p != "accept_and_advance"
        if (not gold_intervene) and pred_intervene:
            over_intervention += 1
        if gold_intervene and (not pred_intervene):
            under_intervention += 1

    n = len(rows)

    result = {
        "num_rows": n,
        "best_action": compute_head_metrics(gold_action, pred_action, ACTION_LABELS),
        "over_intervention_rate": over_intervention / n,
        "under_intervention_rate": under_intervention / n,
    }

    if not action_only:
        gold_local = [r["gold"]["local_correctness"] for r in rows]
        pred_local = [r["pred"]["pred_local_correctness"] for r in rows]

        gold_global = [r["gold"]["global_trajectory"] for r in rows]
        pred_global = [r["pred"]["pred_global_trajectory"] for r in rows]

        gold_issue = [r["gold"]["reasoning_issue"] for r in rows]
        pred_issue = [r["pred"]["pred_reasoning_issue"] for r in rows]

        result["local_correctness"] = compute_head_metrics(gold_local, pred_local, LOCAL_LABELS)
        result["global_trajectory"] = compute_head_metrics(gold_global, pred_global, GLOBAL_LABELS)
        result["reasoning_issue"] = compute_head_metrics(gold_issue, pred_issue, ISSUE_LABELS)

    return result


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--action_only", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit,
        bf16=args.bf16,
    )

    test_rows = load_jsonl(Path(args.test_jsonl))
    if args.limit is not None:
        test_rows = test_rows[: args.limit]

    predictions = []

    pbar = tqdm(
        enumerate(test_rows),
        total=len(test_rows),
        desc="Evaluating policy",
        unit="example",
    )
    for idx, row in pbar:
        if args.action_only:
            pred = predict_action_only(
                model=model,
                tokenizer=tokenizer,
                row=row,
                max_length=args.max_length,
            )
        else:
            pred = predict_structured_labels(
                model=model,
                tokenizer=tokenizer,
                row=row,
                max_length=args.max_length,
            )

        gold = {
            "best_action": row["best_action"],
        }

        if not args.action_only:
            gold.update({
                "local_correctness": row["diagnosis"]["local_correctness"],
                "global_trajectory": row["diagnosis"]["global_trajectory"],
                "reasoning_issue": row["diagnosis"]["reasoning_issue"],
            })

        predictions.append({
            "index": idx,
            "source_problem_id": row.get("source_problem_id"),
            "variant_name": row.get("variant_name"),
            "provenance": row.get("provenance"),
            "teacher_question": row.get("teacher_question"),
            "student_response": row.get("student_response"),
            "gold": gold,
            "pred": pred,
        })

        if (idx + 1) % 50 == 0:
            pbar.set_postfix(processed=f"{idx + 1}/{len(test_rows)}")

    write_jsonl(output_dir / "test_predictions.jsonl", predictions)

    full_eval = {
        "overall": eval_subset(predictions, action_only=args.action_only),
        "by_variant": {
            v: eval_subset([r for r in predictions if r["variant_name"] == v], action_only=args.action_only)
            for v in sorted(set(r["variant_name"] for r in predictions))
        },
        "by_provenance": {
            p: eval_subset([r for r in predictions if r["provenance"] == p], action_only=args.action_only)
            for p in sorted(set(r["provenance"] for r in predictions))
        },
        "config": {
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "test_jsonl": args.test_jsonl,
            "num_test_rows": len(test_rows),
            "max_length": args.max_length,
            "constrained_inference": "candidate_scoring",
            "action_only": args.action_only,
        },
    }

    save_json(output_dir / "eval_metrics.json", full_eval)

    print("\n=== OVERALL ===")
    print("Action accuracy:", full_eval["overall"]["best_action"]["accuracy"])
    print("Action macro-F1:", full_eval["overall"]["best_action"]["macro_f1"])
    print("Over-intervention rate:", full_eval["overall"]["over_intervention_rate"])
    print("Under-intervention rate:", full_eval["overall"]["under_intervention_rate"])
    if not args.action_only:
        print("Local macro-F1:", full_eval["overall"]["local_correctness"]["macro_f1"])
        print("Global macro-F1:", full_eval["overall"]["global_trajectory"]["macro_f1"])
        print("Issue macro-F1:", full_eval["overall"]["reasoning_issue"]["macro_f1"])
    print(f"\nSaved predictions to: {output_dir / 'test_predictions.jsonl'}")
    print(f"Saved metrics to:     {output_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
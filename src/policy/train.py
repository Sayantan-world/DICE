from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =========================================================
# V1 schema decisions
# =========================================================

ACTION_MAP = {
    "accept_and_advance": "accept_and_advance",
    "arithmetic_repair": "arithmetic_repair",
    "handle_question": "handle_question",
    "refocus_subgoal": "refocus_subgoal",
    "stronger_scaffold": "refocus_subgoal",
    "conceptual_hint": "conceptual_hint",
    "ask_for_justification": "ask_for_justification",
    "redirect_irrelevant": None,
    "remediate_to_easier": None,
    "promote_to_harder": None,
}

ISSUE_MAP = {
    "none": "none",
    "arithmetic_error": "arithmetic_error",
    "student_question": "student_question",
    "off_topic": "student_question",
    "subgoal_skip": "subgoal_skip",
    "conceptual_gap": "conceptual_issue",
    "subgoal_confusion": "conceptual_issue",
    "weak_justification": "weak_justification",
}

LOCAL_SET = {"correct", "incorrect", "partial"}
GLOBAL_SET = {"on_track", "ahead", "off_track", "stalled"}
ISSUE_SET = {
    "none",
    "arithmetic_error",
    "student_question",
    "subgoal_skip",
    "conceptual_issue",
    "weak_justification",
}
ACTION_SET = {
    "accept_and_advance",
    "arithmetic_repair",
    "handle_question",
    "refocus_subgoal",
    "conceptual_hint",
    "ask_for_justification",
}

INTERVENTION_ACTIONS = {
    "arithmetic_repair",
    "handle_question",
    "refocus_subgoal",
    "conceptual_hint",
    "ask_for_justification",
}

SYSTEM_PROMPT = """You are a tutoring-policy model.
Given a math tutoring context, diagnose the student's current reasoning state and choose the best next pedagogical action.

Output EXACTLY in this format:
[DIAGNOSIS]
local: <correct|incorrect|partial>
global: <on_track|ahead|off_track|stalled>
issue: <none|arithmetic_error|student_question|subgoal_skip|conceptual_issue|weak_justification>
[ACTION]
<accept_and_advance|arithmetic_repair|handle_question|refocus_subgoal|conceptual_hint|ask_for_justification>
""".strip()

ACTION_ONLY_SYSTEM_PROMPT = """You are a tutoring-policy model.
Given a math tutoring context, choose the best next pedagogical action.

Output EXACTLY in this format:
[ACTION]
<accept_and_advance|arithmetic_repair|handle_question|refocus_subgoal|conceptual_hint|ask_for_justification>
""".strip()


# =========================================================
# Utilities
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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_history(history: Any) -> str:
    if not history:
        return "None"
    out = []
    for turn in history:
        if isinstance(turn, list) and len(turn) == 2:
            out.append(f"Teacher: {turn[0]}\nStudent: {turn[1]}")
    return "\n".join(out) if out else "None"


def build_row_key(row: Dict[str, Any]) -> str:
    if row.get("_row_id"):
        return str(row["_row_id"])

    return "|".join([
        str(row.get("source_problem_id", "")),
        str(row.get("variant_name", "")),
        str(row.get("provenance", "")),
        str(row.get("dialogue_id", "")),
        str(row.get("turn_index", "")),
        str(row.get("step_index", "")),
        str(row.get("synthetic_variant_index", "")),
    ])


def load_iv_lookup(iv_jsonl: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not iv_jsonl:
        return {}

    path = Path(iv_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"IV JSONL not found: {path}")

    iv_rows = load_jsonl(path)
    lookup: Dict[str, Dict[str, Any]] = {}

    duplicate_count = 0
    for row in iv_rows:
        key = row.get("_row_id") or row.get("row_key") or build_row_key(row)
        if key in lookup:
            duplicate_count += 1
        lookup[str(key)] = row

    print(f"Loaded IV rows: {len(iv_rows)}")
    print(f"Loaded unique IV keys: {len(lookup)}")
    print(f"Duplicate IV keys: {duplicate_count}")
    return lookup


def remap_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    row = dict(row)
    diag = row.get("diagnosis", {})
    local = diag.get("local_correctness")
    global_traj = diag.get("global_trajectory")
    issue = diag.get("reasoning_issue")
    action = row.get("best_action")

    action = ACTION_MAP.get(action, None)
    issue = ISSUE_MAP.get(issue, None)

    if action is None:
        return None
    if issue is None:
        return None
    if local not in LOCAL_SET:
        return None
    if global_traj not in GLOBAL_SET:
        return None
    if issue not in ISSUE_SET:
        return None
    if action not in ACTION_SET:
        return None

    row["diagnosis"]["local_correctness"] = local
    row["diagnosis"]["global_trajectory"] = global_traj
    row["diagnosis"]["reasoning_issue"] = issue
    row["best_action"] = action
    return row


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


def format_target(row: Dict[str, Any], action_only: bool = False) -> str:
    if action_only:
        return "[ACTION]\n" + row["best_action"]

    diag = row["diagnosis"]
    return (
        "[DIAGNOSIS]\n"
        f"local: {diag['local_correctness']}\n"
        f"global: {diag['global_trajectory']}\n"
        f"issue: {diag['reasoning_issue']}\n"
        "[ACTION]\n"
        f"{row['best_action']}"
    )


def make_messages(row: Dict[str, Any], action_only: bool = False) -> List[Dict[str, str]]:
    system_prompt = ACTION_ONLY_SYSTEM_PROMPT if action_only else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_user_prompt(row)},
        {"role": "assistant", "content": format_target(row, action_only=action_only)},
    ]


# =========================================================
# Grouped split logic
# =========================================================

def build_group_index(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_problem_id"]].append(row)
    return groups


def compute_overall_label_dist(rows: List[Dict[str, Any]]) -> Dict[str, Counter]:
    stats = {
        "local": Counter(),
        "global": Counter(),
        "issue": Counter(),
        "action": Counter(),
    }
    for r in rows:
        stats["local"][r["diagnosis"]["local_correctness"]] += 1
        stats["global"][r["diagnosis"]["global_trajectory"]] += 1
        stats["issue"][r["diagnosis"]["reasoning_issue"]] += 1
        stats["action"][r["best_action"]] += 1
    return stats


def split_score(
    split_rows: Dict[str, List[Dict[str, Any]]],
    overall: Dict[str, Counter],
    target_sizes: Dict[str, int],
) -> float:
    score = 0.0

    for split_name, rows in split_rows.items():
        n = len(rows)
        score += 3.0 * abs(n - target_sizes[split_name])

        dist = compute_overall_label_dist(rows)

        for head in ["local", "global", "issue", "action"]:
            split_total = sum(dist[head].values())
            overall_total = sum(overall[head].values())
            if split_total == 0:
                score += 1e6
                continue

            labels = set(overall[head].keys())
            for label in labels:
                p_split = dist[head][label] / split_total
                p_all = overall[head][label] / overall_total
                score += 100.0 * abs(p_split - p_all)

            if split_name in {"val", "test"}:
                for label, count in overall[head].items():
                    if count >= 20 and dist[head][label] == 0:
                        score += 1000.0

    return score


def grouped_random_search_split(
    rows: List[Dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_trials: int = 200,
) -> Dict[str, List[Dict[str, Any]]]:
    groups = build_group_index(rows)
    group_ids = list(groups.keys())
    n_total = len(rows)
    target_sizes = {
        "train": int(round(n_total * train_ratio)),
        "val": int(round(n_total * val_ratio)),
        "test": n_total - int(round(n_total * train_ratio)) - int(round(n_total * val_ratio)),
    }

    overall = compute_overall_label_dist(rows)

    best_score = float("inf")
    best_split = None

    for trial in range(n_trials):
        rng_trial = random.Random(seed + trial)
        shuffled = group_ids[:]
        rng_trial.shuffle(shuffled)

        split_rows = {"train": [], "val": [], "test": []}
        split_sizes = {"train": 0, "val": 0, "test": 0}

        for gid in shuffled:
            group_rows = groups[gid]
            gsize = len(group_rows)

            deficits = {
                k: target_sizes[k] - split_sizes[k]
                for k in split_sizes
            }

            candidate_order = sorted(deficits.keys(), key=lambda k: deficits[k], reverse=True)

            assigned = False
            for split_name in candidate_order:
                if split_sizes[split_name] < target_sizes[split_name] or split_name == candidate_order[0]:
                    split_rows[split_name].extend(group_rows)
                    split_sizes[split_name] += gsize
                    assigned = True
                    break

            if not assigned:
                split_rows["train"].extend(group_rows)
                split_sizes["train"] += gsize

        score = split_score(split_rows, overall, target_sizes)
        if score < best_score:
            best_score = score
            best_split = split_rows

    if best_split is None:
        raise RuntimeError("Failed to create grouped split.")

    return best_split


def print_split_summary(split_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    print("\n=== Split Summary ===")
    for split_name, rows in split_rows.items():
        stats = compute_overall_label_dist(rows)
        print(f"\n[{split_name}] rows={len(rows)} problems={len(set(r['source_problem_id'] for r in rows))}")
        for head in ["local", "global", "issue", "action"]:
            print(f"  {head}: {dict(stats[head])}")


# =========================================================
# Dataset prep
# =========================================================

def prepare_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for row in rows:
        mapped = remap_row(row)
        if mapped is not None:
            cleaned.append(mapped)
    return cleaned


def compute_sample_weight(
    row: Dict[str, Any],
    iv_lookup: Dict[str, Dict[str, Any]],
    use_iv_weighting: bool,
    iv_apply_variant: str,
    iv_apply_provenance: str,
    missing_iv_weight: float,
    accept_weight: float,
    min_iv_weight: float,
    max_iv_weight: float,
) -> Tuple[float, Optional[float], bool]:
    """
    Returns:
      sample_weight, intervention_value, has_iv
    """

    if not use_iv_weighting:
        return 1.0, None, False

    action = row.get("best_action")
    variant = row.get("variant_name")
    provenance = row.get("provenance")

    if action == "accept_and_advance":
        return float(accept_weight), None, False

    should_apply = (
        action in INTERVENTION_ACTIONS
        and variant == iv_apply_variant
        and provenance == iv_apply_provenance
    )

    if not should_apply:
        return 1.0, None, False

    key = build_row_key(row)
    iv_row = iv_lookup.get(key)

    if iv_row is None:
        return float(missing_iv_weight), None, False

    iv = float(iv_row.get("intervention_value", 0.0))
    weight = max(iv, 0.0)
    weight = max(weight, min_iv_weight)
    weight = min(weight, max_iv_weight)

    return float(weight), iv, True


def build_examples(
    rows: List[Dict[str, Any]],
    action_only: bool = False,
    iv_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    use_iv_weighting: bool = False,
    iv_apply_variant: str = "target",
    iv_apply_provenance: str = "observed_multi_dialogue",
    missing_iv_weight: float = 1.0,
    accept_weight: float = 1.0,
    min_iv_weight: float = 0.0,
    max_iv_weight: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    iv_lookup = iv_lookup or {}
    examples = []

    stats = {
        "num_rows": 0,
        "num_weighted_rows": 0,
        "num_missing_iv_rows": 0,
        "num_accept_rows": 0,
        "num_non_target_or_non_observed_rows": 0,
        "weight_sum": 0.0,
        "weight_min": None,
        "weight_max": None,
        "by_action": Counter(),
        "weighted_by_action": Counter(),
        "missing_iv_by_action": Counter(),
    }

    for row in rows:
        weight, iv, has_iv = compute_sample_weight(
            row=row,
            iv_lookup=iv_lookup,
            use_iv_weighting=use_iv_weighting,
            iv_apply_variant=iv_apply_variant,
            iv_apply_provenance=iv_apply_provenance,
            missing_iv_weight=missing_iv_weight,
            accept_weight=accept_weight,
            min_iv_weight=min_iv_weight,
            max_iv_weight=max_iv_weight,
        )

        action = row.get("best_action")
        variant = row.get("variant_name")
        provenance = row.get("provenance")

        stats["num_rows"] += 1
        stats["weight_sum"] += weight
        stats["by_action"][action] += 1

        if stats["weight_min"] is None or weight < stats["weight_min"]:
            stats["weight_min"] = weight
        if stats["weight_max"] is None or weight > stats["weight_max"]:
            stats["weight_max"] = weight

        if action == "accept_and_advance":
            stats["num_accept_rows"] += 1

        if not (variant == iv_apply_variant and provenance == iv_apply_provenance):
            stats["num_non_target_or_non_observed_rows"] += 1

        if use_iv_weighting and action in INTERVENTION_ACTIONS and variant == iv_apply_variant and provenance == iv_apply_provenance:
            if has_iv:
                stats["num_weighted_rows"] += 1
                stats["weighted_by_action"][action] += 1
            else:
                stats["num_missing_iv_rows"] += 1
                stats["missing_iv_by_action"][action] += 1

        examples.append({
            "source_problem_id": row["source_problem_id"],
            "row_key": build_row_key(row),
            "variant_name": row.get("variant_name", ""),
            "provenance": row.get("provenance", ""),
            "best_action": row["best_action"],
            "intervention_value": iv if iv is not None else 0.0,
            "sample_weight": weight,
            "text_input": format_user_prompt(row),
            "text_target": format_target(row, action_only=action_only),
            "messages": make_messages(row, action_only=action_only),
        })

    stats["mean_weight"] = stats["weight_sum"] / max(stats["num_rows"], 1)
    stats["by_action"] = dict(stats["by_action"])
    stats["weighted_by_action"] = dict(stats["weighted_by_action"])
    stats["missing_iv_by_action"] = dict(stats["missing_iv_by_action"])
    return examples, stats


def to_hf(examples: List[Dict[str, Any]], action_only: bool = False) -> Dataset:
    system_prompt = ACTION_ONLY_SYSTEM_PROMPT if action_only else SYSTEM_PROMPT
    rows = []
    for ex in examples:
        rows.append({
            "source_problem_id": ex["source_problem_id"],
            "row_key": ex["row_key"],
            "variant_name": ex["variant_name"],
            "provenance": ex["provenance"],
            "best_action": ex["best_action"],
            "intervention_value": float(ex["intervention_value"]),
            "sample_weight": float(ex["sample_weight"]),
            "prompt_messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["text_input"]},
            ],
            "full_messages": ex["messages"],
            "target_text": ex["text_target"],
        })
    return Dataset.from_list(rows)


# =========================================================
# Tokenization / collator
# =========================================================

@dataclass
class DataCollatorForCausalLMWithMask:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_batch = []
        labels_batch = []
        attention_masks = []
        sample_weights = []

        for feat in features:
            prompt = self.tokenizer.apply_chat_template(
                feat["prompt_messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            full = self.tokenizer.apply_chat_template(
                feat["full_messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(
                full,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]

            labels = full_ids[:]
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels[:prompt_len] = [-100] * prompt_len

            attn = [1] * len(full_ids)

            input_ids_batch.append(torch.tensor(full_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            attention_masks.append(torch.tensor(attn, dtype=torch.long))
            sample_weights.append(float(feat.get("sample_weight", 1.0)))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_batch,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_batch,
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float),
        }


# =========================================================
# IV + KL Trainer
# =========================================================

class IVKLTrainer(Trainer):
    def __init__(
        self,
        *args,
        use_iv_weighting: bool = False,
        use_kl_regularization: bool = False,
        kl_beta: float = 0.05,
        ref_model: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_iv_weighting = use_iv_weighting
        self.use_kl_regularization = use_kl_regularization
        self.kl_beta = kl_beta
        self.ref_model = ref_model

        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_weight = inputs.pop("sample_weight", None)

        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        token_loss = token_loss.view(shift_labels.shape)

        mask = (shift_labels != -100).float()
        per_example_ce = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        # Paper Eq. (2), minibatch form:
        #   L = -(1/N) sum_i w_i log π(a_i | x_i) + (β/N) sum_i KL(π(·|x_i) || π_SFT(·|x_i))
        # CE is sample-count normalized (not divided by sum of weights). KL is an
        # unweighted mean over the same action tokens (--action_only).
        if self.use_iv_weighting and sample_weight is not None:
            sample_weight = sample_weight.to(per_example_ce.device).to(per_example_ce.dtype)
            ce_loss = (sample_weight * per_example_ce).mean()
        else:
            ce_loss = per_example_ce.mean()

        kl_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)

        if self.use_kl_regularization:
            if self.ref_model is None:
                raise RuntimeError("KL regularization is enabled but ref_model is None.")

            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                ref_logits = ref_outputs.logits

            ref_shift_logits = ref_logits[..., :-1, :].contiguous()

            cur_log_probs = F.log_softmax(shift_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_shift_logits, dim=-1)
            cur_probs = F.softmax(shift_logits, dim=-1)

            kl_token = (cur_probs * (cur_log_probs - ref_log_probs)).sum(dim=-1)

            kl_per_example = (kl_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            kl_loss = kl_per_example.mean()

        loss = ce_loss + self.kl_beta * kl_loss

        if self.state.global_step % 20 == 0 and model.training:
            try:
                self.log({
                    "train/ce_loss": ce_loss.detach().float().item(),
                    "train/kl_loss": kl_loss.detach().float().item(),
                    "train/kl_beta": float(self.kl_beta),
                    "train/total_loss": loss.detach().float().item(),
                })
            except Exception:
                pass

        return (loss, outputs) if return_outputs else loss


# =========================================================
# Model loading
# =========================================================

def load_base_model(
    model_name: str,
    use_4bit: bool,
    bf16: bool,
    gradient_checkpointing: bool = False,
):
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model


def attach_lora_for_training(
    model,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    return model


def load_reference_model(
    model_name: str,
    ref_adapter_path: Optional[str],
    use_4bit: bool,
    bf16: bool,
):
    ref_model = load_base_model(
        model_name=model_name,
        use_4bit=use_4bit,
        bf16=bf16,
        gradient_checkpointing=False,
    )

    if ref_adapter_path:
        print(f"Loading KL reference adapter from: {ref_adapter_path}")
        ref_model = PeftModel.from_pretrained(ref_model, ref_adapter_path)

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    return ref_model


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Path to cleaned decision-point JSONL.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Optional directory for train/val/test split files. If existing files are found, splitting is skipped.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=2048)

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--action_only", action="store_true")

    # IV weighting
    parser.add_argument("--use_iv_weighting", action="store_true")
    parser.add_argument(
        "--iv_jsonl",
        type=str,
        default=None,
        help="Train-split IV JSONL. Should contain _row_id or row_key and intervention_value.",
    )
    parser.add_argument("--iv_apply_variant", type=str, default="target")
    parser.add_argument("--iv_apply_provenance", type=str, default="observed_multi_dialogue")
    parser.add_argument(
        "--missing_iv_weight",
        type=float,
        default=1.0,
        help="Weight for target intervention rows missing IV. Use 1.0 for safe fallback.",
    )
    parser.add_argument(
        "--accept_weight",
        type=float,
        default=1.0,
        help="Weight for accept_and_advance rows.",
    )
    parser.add_argument(
        "--min_iv_weight",
        type=float,
        default=0.0,
        help="Minimum IV weight after clamp.",
    )
    parser.add_argument(
        "--max_iv_weight",
        type=float,
        default=1.0,
        help="Maximum IV weight after clamp.",
    )

    # KL regularization
    parser.add_argument("--use_kl_regularization", action="store_true")
    parser.add_argument("--kl_beta", type=float, default=0.05)
    parser.add_argument(
        "--kl_ref_adapter_path",
        type=str,
        default=None,
        help="LoRA adapter for π_SFT in Eq. (2). Required when --use_kl_regularization is set.",
    )

    # reporting
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default="DICE")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")

    args = parser.parse_args()

    set_seed(args.seed)

    if not args.input_jsonl:
        raise ValueError("--input_jsonl is required (cleaned decision-point JSONL).")

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if not os.getenv("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is not set. Export it before running training.")

    input_path = Path(args.input_jsonl)
    if args.split_dir is None:
        split_dir = input_path.parent / "splits_v1_seed42"
    else:
        split_dir = Path(args.split_dir)

    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "val.jsonl"
    test_path = split_dir / "test.jsonl"
    split_meta_path = split_dir / "split_meta.json"

    # -----------------------------------------------------
    # Create or reuse splits
    # -----------------------------------------------------
    if train_path.exists() and val_path.exists() and test_path.exists():
        print(f"Found existing split files in {split_dir}. Skipping split creation.")
        train_rows = load_jsonl(train_path)
        val_rows = load_jsonl(val_path)
        test_rows = load_jsonl(test_path)
    else:
        print("Creating grouped data splits...")
        all_rows = load_jsonl(input_path)
        all_rows = prepare_records(all_rows)

        split_rows = grouped_random_search_split(
            all_rows,
            seed=args.seed,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            n_trials=200,
        )

        print_split_summary(split_rows)

        split_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(train_path, split_rows["train"])
        write_jsonl(val_path, split_rows["val"])
        write_jsonl(test_path, split_rows["test"])

        meta = {
            "seed": args.seed,
            "train_rows": len(split_rows["train"]),
            "val_rows": len(split_rows["val"]),
            "test_rows": len(split_rows["test"]),
            "train_problems": len(set(r["source_problem_id"] for r in split_rows["train"])),
            "val_problems": len(set(r["source_problem_id"] for r in split_rows["val"])),
            "test_problems": len(set(r["source_problem_id"] for r in split_rows["test"])),
            "v1_action_set": sorted(ACTION_SET),
            "v1_issue_set": sorted(ISSUE_SET),
        }
        save_json(split_meta_path, meta)

        train_rows = split_rows["train"]
        val_rows = split_rows["val"]
        test_rows = split_rows["test"]

    # -----------------------------------------------------
    # IV lookup
    # -----------------------------------------------------
    iv_lookup = load_iv_lookup(args.iv_jsonl) if args.use_iv_weighting else {}

    # -----------------------------------------------------
    # Build examples
    # -----------------------------------------------------
    train_examples, train_weight_stats = build_examples(
        train_rows,
        action_only=args.action_only,
        iv_lookup=iv_lookup,
        use_iv_weighting=args.use_iv_weighting,
        iv_apply_variant=args.iv_apply_variant,
        iv_apply_provenance=args.iv_apply_provenance,
        missing_iv_weight=args.missing_iv_weight,
        accept_weight=args.accept_weight,
        min_iv_weight=args.min_iv_weight,
        max_iv_weight=args.max_iv_weight,
    )

    val_examples, val_weight_stats = build_examples(
        val_rows,
        action_only=args.action_only,
        iv_lookup={},
        use_iv_weighting=False,
    )

    test_examples, test_weight_stats = build_examples(
        test_rows,
        action_only=args.action_only,
        iv_lookup={},
        use_iv_weighting=False,
    )

    print("\n=== IV Weighting Stats: Train ===")
    print(json.dumps(train_weight_stats, indent=2, ensure_ascii=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "train_weight_stats.json", train_weight_stats)

    ds = DatasetDict({
        "train": to_hf(train_examples, action_only=args.action_only),
        "validation": to_hf(val_examples, action_only=args.action_only),
        "test": to_hf(test_examples, action_only=args.action_only),
    })

    # -----------------------------------------------------
    # Tokenizer / model
    # -----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_base_model(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    model = attach_lora_for_training(
        model=model,
        use_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    model.print_trainable_parameters()

    ref_model = None
    if args.use_kl_regularization:
        if not args.kl_ref_adapter_path:
            raise ValueError(
                "--use_kl_regularization requires --kl_ref_adapter_path (the SFT policy π_SFT in Eq. 2)."
            )
        ref_model = load_reference_model(
            model_name=args.model_name,
            ref_adapter_path=args.kl_ref_adapter_path,
            use_4bit=args.use_4bit,
            bf16=args.bf16,
        )

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------
    collator = DataCollatorForCausalLMWithMask(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name or Path(args.output_dir).name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = IVKLTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        callbacks=callbacks,
        use_iv_weighting=args.use_iv_weighting,
        use_kl_regularization=args.use_kl_regularization,
        kl_beta=args.kl_beta,
        ref_model=ref_model,
    )

    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    ds["test"].to_json(str(Path(args.output_dir) / "test_dataset.jsonl"))

    config = vars(args)
    save_json(Path(args.output_dir) / "training_config.json", config)

    print(f"Done. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
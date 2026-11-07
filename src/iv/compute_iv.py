from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from iv_prompts import (
    ACTION_SET,
    JUDGE_LABELS,
    JUDGE_SCORE_MAP,
    JUDGE_SYSTEM,
    STUDENT_SIM_SYSTEM,
    build_judge_user,
    build_student_sim_user,
    teacher_template,
)


# ----------------------------
# Local utilities
# ----------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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


def load_existing_processed_keys(path: Path) -> set[str]:
    existing = load_jsonl(path)
    keys = set()
    for row in existing:
        key = row.get("_row_id") or row.get("row_key")
        if key:
            keys.add(str(key))
    return keys


def load_model_and_tokenizer(
    model_name: str,
    adapter_path: Optional[str] = None,
    use_4bit: bool = False,
    bf16: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
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
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def render_chat(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


# ----------------------------
# NVIDIA judge client
# ----------------------------

def load_nvidia_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing NVIDIA_API_KEY in environment/.env.")
    return OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
    )


def extract_text_from_chat_completion(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("No choices in response.")

        message = choices[0].message
        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            chunks = []
            for block in content:
                if isinstance(block, dict):
                    if "text" in block and isinstance(block["text"], str):
                        chunks.append(block["text"])
                    elif block.get("type") == "text" and isinstance(block.get("text"), str):
                        chunks.append(block["text"])
                else:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        chunks.append(text)
            joined = "".join(chunks).strip()
            if joined:
                return joined

        reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return reasoning_content.strip()

        reasoning = getattr(message, "reasoning", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        maybe_str = str(content).strip() if content is not None else ""
        if maybe_str and maybe_str.lower() != "none":
            return maybe_str

        raise ValueError("Empty message.content")
    except Exception as e:
        raise ValueError(
            f"Could not extract text from NVIDIA chat completion response. "
            f"Response type={type(response)}"
        ) from e


def extract_judge_label(raw_text: str) -> Optional[str]:
    text = raw_text.strip().lower()
    if not text:
        return None

    if text in JUDGE_LABELS:
        return text

    first = text.split()[0].strip(".,:;!?\"'`()[]{}")
    if first in JUDGE_LABELS:
        return first

    for label in JUDGE_LABELS:
        if label in text:
            return label

    if "helpful" in text:
        return "helpful_progress"
    if "partial" in text:
        return "partial_progress"
    if "harm" in text or "no_progress" in text or "no progress" in text:
        return "no_progress_or_harm"
    return None


class NvidiaRateLimiter:
    def __init__(self, min_interval_seconds: float = 2.0) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._last_call_time: Optional[float] = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last_call_time is None:
            self._last_call_time = now
            return
        elapsed = now - self._last_call_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_call_time = time.monotonic()


def judge_progress_with_nvidia(
    client: OpenAI,
    rate_limiter: NvidiaRateLimiter,
    judge_model_names: List[str],
    row: Dict[str, Any],
    action: str,
    teacher_utterance: str,
    simulated_student_reply: str,
    max_retries_per_model: int = 2,
) -> Tuple[str, float, Dict[str, float], str, str]:
    """
    Returns:
      label, score, score_breakdown, raw_text, model_used
    """
    user_prompt = build_judge_user(
        row=row,
        action=action,
        teacher_utterance=teacher_utterance,
        simulated_student_reply=simulated_student_reply,
    )

    last_err: Optional[Exception] = None

    for model_name in judge_model_names:
        for attempt in range(max_retries_per_model):
            try:
                rate_limiter.wait()
                token_budget = 256
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"{user_prompt}\n\n"
                                "IMPORTANT: Output the label as the very first token, with no explanation."
                            ),
                        },
                    ],
                    max_tokens=token_budget,
                )

                raw_text = extract_text_from_chat_completion(response).strip()
                normalized = extract_judge_label(raw_text)

                if normalized is None or normalized not in JUDGE_LABELS:
                    raise ValueError(f"Judge returned invalid label: {raw_text}")

                score_breakdown = {label: 1.0 if label == normalized else 0.0 for label in JUDGE_LABELS}
                return normalized, JUDGE_SCORE_MAP[normalized], score_breakdown, raw_text, model_name

            except Exception as e:
                last_err = e
                if attempt < max_retries_per_model - 1:
                    time.sleep(1.5 * (attempt + 1))
                else:
                    # switch to next fallback model
                    pass

    raise RuntimeError(f"All NVIDIA judge models failed. Last error: {last_err}") from last_err


# ----------------------------
# Simulation + IV
# ----------------------------

def simulate_next_student_reply(
    student_model,
    student_tokenizer,
    row: Dict[str, Any],
    teacher_utterance: str,
    seed: int,
    max_new_tokens: int = 80,
) -> str:
    user_prompt = build_student_sim_user(row=row, teacher_utterance=teacher_utterance)
    prompt_text = render_chat(student_tokenizer, STUDENT_SIM_SYSTEM, user_prompt)

    reply = generate_text(
        model=student_model,
        tokenizer=student_tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        seed=seed,
    )
    return reply.strip()


def compute_mu_for_action(
    row: Dict[str, Any],
    action: str,
    student_model,
    student_tokenizer,
    judge_client: OpenAI,
    judge_model_names: List[str],
    judge_rate_limiter: NvidiaRateLimiter,
    k_rollouts: int,
    base_seed: int,
) -> Dict[str, Any]:
    utterance = teacher_template(action, row)

    rollout_records = []
    scores = []
    judge_label_counter = Counter()
    judge_model_counter = Counter()

    for k in range(k_rollouts):
        seed = base_seed + k
        student_reply = simulate_next_student_reply(
            student_model=student_model,
            student_tokenizer=student_tokenizer,
            row=row,
            teacher_utterance=utterance,
            seed=seed,
        )

        judge_label, judge_score, judge_score_breakdown, raw_judge_text, judge_model_used = judge_progress_with_nvidia(
            client=judge_client,
            rate_limiter=judge_rate_limiter,
            judge_model_names=judge_model_names,
            row=row,
            action=action,
            teacher_utterance=utterance,
            simulated_student_reply=student_reply,
        )

        judge_label_counter[judge_label] += 1
        judge_model_counter[judge_model_used] += 1

        rollout_records.append({
            "rollout_index": k,
            "teacher_utterance": utterance,
            "simulated_student_reply": student_reply,
            "judge_label": judge_label,
            "judge_score": judge_score,
            "judge_score_breakdown": judge_score_breakdown,
            "judge_raw_text": raw_judge_text,
            "judge_model_used": judge_model_used,
        })
        scores.append(judge_score)

    mu_hat = sum(scores) / len(scores) if scores else 0.0

    return {
        "action": action,
        "teacher_utterance": utterance,
        "mu_hat": mu_hat,
        "scores": scores,
        "judge_label_counts": dict(judge_label_counter),
        "judge_model_counts": dict(judge_model_counter),
        "rollouts": rollout_records,
    }


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--stats_json", type=str, required=True)
    parser.add_argument("--errors_jsonl", type=str, default=None)

    parser.add_argument("--student_model_name", type=str, required=True)
    parser.add_argument("--student_adapter_path", type=str, default=None)

    parser.add_argument(
        "--judge_model_names",
        type=str,
        default="moonshotai/kimi-k2-instruct-0905,openai/gpt-oss-120b,meta/llama-3.3-70b-instruct",
        help="Comma-separated NVIDIA API model ids in fallback order.",
    )

    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--k_rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--filter_variant", type=str, default="target")
    parser.add_argument("--filter_provenance", type=str, default="observed_multi_dialogue")
    parser.add_argument(
        "--min_api_interval_seconds",
        type=float,
        default=2.0,
        help="At least 2.0 sec to stay within 30 RPM.",
    )
    args = parser.parse_args()

    judge_model_names = [x.strip() for x in args.judge_model_names.split(",") if x.strip()]
    if not judge_model_names:
        raise RuntimeError("No judge models provided.")

    rows = load_jsonl(Path(args.input_jsonl))

    filtered = []
    for row in rows:
        if args.filter_variant and row.get("variant_name") != args.filter_variant:
            continue
        if args.filter_provenance and row.get("provenance") != args.filter_provenance:
            continue
        filtered.append(row)

    if args.limit is not None:
        filtered = filtered[: args.limit]

    output_path = Path(args.output_jsonl)
    stats_path = Path(args.stats_json)
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else None

    processed_keys = load_existing_processed_keys(output_path)
    filtered_to_run = []
    for row in filtered:
        row_key = build_row_key(row)
        if row_key in processed_keys:
            continue
        filtered_to_run.append(row)

    print(f"Loaded {len(filtered)} rows after filtering.")
    print(f"Already processed: {len(processed_keys)}")
    print(f"Remaining to run:  {len(filtered_to_run)}")

    student_model, student_tokenizer = load_model_and_tokenizer(
        model_name=args.student_model_name,
        adapter_path=args.student_adapter_path,
        use_4bit=args.use_4bit,
        bf16=args.bf16,
    )

    judge_client = load_nvidia_client()
    judge_rate_limiter = NvidiaRateLimiter(min_interval_seconds=args.min_api_interval_seconds)

    # Load existing records for final stats
    all_records = load_jsonl(output_path)
    overall_judge_counter = Counter()
    action_arm_counter = Counter()
    accept_arm_counter = Counter()
    judge_model_counter_overall = Counter()

    for record in all_records:
        for label, count in record.get("action_eval", {}).get("judge_label_counts", {}).items():
            overall_judge_counter[label] += count
            action_arm_counter[label] += count
        for label, count in record.get("accept_eval", {}).get("judge_label_counts", {}).items():
            overall_judge_counter[label] += count
            accept_arm_counter[label] += count
        for model_name, count in record.get("action_eval", {}).get("judge_model_counts", {}).items():
            judge_model_counter_overall[model_name] += count
        for model_name, count in record.get("accept_eval", {}).get("judge_model_counts", {}).items():
            judge_model_counter_overall[model_name] += count

    pbar = tqdm(
        enumerate(filtered_to_run),
        total=len(filtered_to_run),
        desc="Computing intervention value",
        unit="row",
    )

    for idx, row in pbar:
        chosen_action = row["best_action"]
        if chosen_action not in ACTION_SET:
            continue

        row_key = build_row_key(row)
        base_seed = args.seed + idx * 1000

        try:
            action_result = compute_mu_for_action(
                row=row,
                action=chosen_action,
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                judge_client=judge_client,
                judge_model_names=judge_model_names,
                judge_rate_limiter=judge_rate_limiter,
                k_rollouts=args.k_rollouts,
                base_seed=base_seed,
            )

            accept_result = compute_mu_for_action(
                row=row,
                action="accept_and_advance",
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                judge_client=judge_client,
                judge_model_names=judge_model_names,
                judge_rate_limiter=judge_rate_limiter,
                k_rollouts=args.k_rollouts,
                base_seed=base_seed + 100,
            )

            for label, count in action_result["judge_label_counts"].items():
                overall_judge_counter[label] += count
                action_arm_counter[label] += count
            for label, count in accept_result["judge_label_counts"].items():
                overall_judge_counter[label] += count
                accept_arm_counter[label] += count
            for model_name, count in action_result.get("judge_model_counts", {}).items():
                judge_model_counter_overall[model_name] += count
            for model_name, count in accept_result.get("judge_model_counts", {}).items():
                judge_model_counter_overall[model_name] += count

            iv = action_result["mu_hat"] - accept_result["mu_hat"]

            record = {
                "index": len(all_records),
                "_row_id": row.get("_row_id"),
                "row_key": row_key,
                "source_problem_id": row.get("source_problem_id"),
                "variant_name": row.get("variant_name"),
                "provenance": row.get("provenance"),
                "dialogue_id": row.get("dialogue_id"),
                "turn_index": row.get("turn_index"),
                "step_index": row.get("step_index"),
                "synthetic_variant_index": row.get("synthetic_variant_index"),
                "teacher_question": row.get("teacher_question"),
                "student_response": row.get("student_response"),
                "gold_action": chosen_action,
                "mu_action": action_result["mu_hat"],
                "mu_accept_and_advance": accept_result["mu_hat"],
                "intervention_value": iv,
                "action_eval": action_result,
                "accept_eval": accept_result,
            }

            append_jsonl(output_path, record)
            all_records.append(record)

            if (len(all_records)) % 5 == 0:
                mean_iv_so_far = sum(r["intervention_value"] for r in all_records) / len(all_records)
                pbar.set_postfix(
                    saved=len(all_records),
                    mean_iv=f"{mean_iv_so_far:.4f}",
                )

        except Exception as e:
            error_record = {
                "_row_id": row.get("_row_id"),
                "row_key": row_key,
                "source_problem_id": row.get("source_problem_id"),
                "variant_name": row.get("variant_name"),
                "provenance": row.get("provenance"),
                "dialogue_id": row.get("dialogue_id"),
                "turn_index": row.get("turn_index"),
                "step_index": row.get("step_index"),
                "synthetic_variant_index": row.get("synthetic_variant_index"),
                "teacher_question": row.get("teacher_question"),
                "student_response": row.get("student_response"),
                "gold_action": chosen_action,
                "error": str(e),
            }
            if errors_path is not None:
                append_jsonl(errors_path, error_record)
            continue

    stats = {
        "num_rows": 0,
        "mean_mu_action": 0.0,
        "mean_mu_accept": 0.0,
        "mean_intervention_value": 0.0,
        "num_positive_iv": 0,
        "num_zeroish_iv": 0,
        "num_negative_iv": 0,
        "judge_label_distribution_overall": {},
        "judge_label_distribution_action_arm": {},
        "judge_label_distribution_accept_arm": {},
        "judge_model_usage_overall": {},
        "config": {
            "student_model_name": args.student_model_name,
            "student_adapter_path": args.student_adapter_path,
            "judge_model_names": judge_model_names,
            "k_rollouts": args.k_rollouts,
            "seed": args.seed,
            "filter_variant": args.filter_variant,
            "filter_provenance": args.filter_provenance,
            "min_api_interval_seconds": args.min_api_interval_seconds,
        },
    }

    if all_records:
        mean_mu_action = sum(r["mu_action"] for r in all_records) / len(all_records)
        mean_mu_accept = sum(r["mu_accept_and_advance"] for r in all_records) / len(all_records)
        mean_iv = sum(r["intervention_value"] for r in all_records) / len(all_records)

        delta = 1e-6
        num_pos = sum(1 for r in all_records if r["intervention_value"] > delta)
        num_neg = sum(1 for r in all_records if r["intervention_value"] < -delta)
        num_zeroish = len(all_records) - num_pos - num_neg

        stats.update({
            "num_rows": len(all_records),
            "mean_mu_action": mean_mu_action,
            "mean_mu_accept": mean_mu_accept,
            "mean_intervention_value": mean_iv,
            "num_positive_iv": num_pos,
            "num_zeroish_iv": num_zeroish,
            "num_negative_iv": num_neg,
            "judge_label_distribution_overall": dict(overall_judge_counter),
            "judge_label_distribution_action_arm": dict(action_arm_counter),
            "judge_label_distribution_accept_arm": dict(accept_arm_counter),
            "judge_model_usage_overall": dict(judge_model_counter_overall),
        })

    save_json(stats_path, stats)
    print(f"Saved IV records to: {output_path}")
    print(f"Saved stats to:      {stats_path}")
    if errors_path is not None:
        print(f"Saved errors to:     {errors_path}")


if __name__ == "__main__":
    main()
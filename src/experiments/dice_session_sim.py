#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Put this file in the same directory as dice_policy_sim.py
import dice_policy_sim as dice_mod

from dice_policy_sim import (
    ACTIVE_ACTIONS,
    NON_INTERVENTION_ACTION,
    ACTION_DESCRIPTIONS,
    DEFAULT_INPUT,
    DEFAULT_PRED_ROOT,
    DEFAULT_RES_ROOT,
    TEACHER_MODEL,
    STUDENT_MODEL,
    POLICY_BASE_MODEL,
    model_slug,
    load_json,
    append_jsonl,
    mean_or_zero,
    build_llm,
    generate_one_vllm,
    generate_one_hf,
    build_student_prompt_text,
    is_correct,
    teacher_accepts,
    load_policy_model_and_tokenizer,
)

SESSION_VARIANTS = ["easy", "target", "hard"]
VARIANT_DISPLAY = {"easy": "easy", "target": "source", "hard": "hard"}


def looks_like_final_answer(student_text: str) -> bool:
    """Conservative final-answer detector for unsafe-accept diagnostics/guard."""
    t = str(student_text).lower()
    explicit_markers = [
        "final answer", "final total", "answer:", "answer is", "the answer is",
        "so the answer", "therefore the answer", "therefore, the answer", "the final answer",
    ]
    if any(m in t for m in explicit_markers):
        return True
    if re.search(r"\bthe fee first exceeds\b", t):
        return True
    if re.search(r"\bthe total cost (is|would be)\b", t):
        return True
    if re.search(r"\btotal daily salary (is|=)\b", t):
        return True
    nums = re.findall(r"[-+]?\d*\.?\d+", t.replace(",", ""))
    return bool(nums) and len(t.split()) <= 8


def classify_acceptance_event(
    teacher_text: str,
    last_student_text: str,
    last_student_is_correct: bool,
    teacher_accepted_override: Optional[bool] = None,
) -> Dict[str, Any]:
    """Use text-based acceptance for response-only; use action-based override for DICE."""
    accepted = teacher_accepts(teacher_text) if teacher_accepted_override is None else bool(teacher_accepted_override)
    student_looked_final = looks_like_final_answer(last_student_text)
    if not accepted:
        return {
            "teacher_accepted": False,
            "conservative_false_acceptance": False,
            "unsafe_final_acceptance": False,
            "step_acceptance_after_nonfinal": False,
            "student_looked_final": student_looked_final,
        }
    if last_student_is_correct:
        return {
            "teacher_accepted": True,
            "conservative_false_acceptance": False,
            "unsafe_final_acceptance": False,
            "step_acceptance_after_nonfinal": False,
            "student_looked_final": student_looked_final,
        }
    return {
        "teacher_accepted": True,
        "conservative_false_acceptance": True,
        "unsafe_final_acceptance": student_looked_final,
        "step_acceptance_after_nonfinal": not student_looked_final,
        "student_looked_final": student_looked_final,
    }


def is_socraticlm_model(model_name: str) -> bool:
    return "socraticlm" in model_name.lower()


def build_socraticlm_teacher_conversation(
    question: str,
    answer: Any,
    steps: Any,
    dialogue: List[Dict[str, Any]],
    action: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Socratic teacher, please guide me to solve the [Problem] "
                "with heuristic questions based on the following information.\n"
            ),
        },
        {"role": "user", "content": f"[Problem] {question} [Answer] {answer} [Analysis] {steps}"},
    ]
    if not dialogue:
        messages.append({
            "role": "user",
            "content": "Please start tutoring me. Ask one concise Socratic question first. Do not reveal the final answer immediately.",
        })
    else:
        for msg in dialogue:
            role = "assistant" if msg["speaker"] == "teacher" else "user"
            messages.append({"role": role, "content": msg["text"]})
        if action is not None:
            messages.append({
                "role": "user",
                "content": (
                    f"The selected pedagogical action is: {action}. "
                    f"Action meaning: {ACTION_DESCRIPTIONS[action]} "
                    "Generate the next teacher utterance in a concise Socratic style. "
                    "Follow the selected action and do not reveal unnecessary solution steps."
                ),
            })
    return messages


def build_generic_response_only_teacher_conversation(
    question: str,
    dialogue: List[Dict[str, Any]],
    answer: Any | None = None,
    steps: Any | None = None,
    use_gold_context: bool = False,
) -> List[Dict[str, str]]:
    system = (
        "You are a Socratic math tutor. Help the student solve the problem through concise guidance. "
        "Ask at most one question or give one short hint at a time. "
        "Do not reveal the final answer unless the student has already solved it."
    )
    user_seed = f"[Problem]\n{question}\n"
    if use_gold_context:
        user_seed += f"\n[Gold Answer]\n{answer}\n\n[Reference Steps]\n{steps}\n"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_seed}]
    if not dialogue:
        messages.append({"role": "user", "content": "Please start tutoring me. Ask one concise Socratic question first. Do not reveal the final answer immediately."})
    else:
        for msg in dialogue:
            role = "assistant" if msg["speaker"] == "teacher" else "user"
            messages.append({"role": role, "content": msg["text"]})
        messages.append({"role": "user", "content": "Generate the next teacher utterance in a Socratic style. Be concise and do not reveal unnecessary solution steps."})
    return messages


def build_response_only_teacher_conversation(
    teacher_model_name: str,
    question: str,
    dialogue: List[Dict[str, Any]],
    answer: Any | None = None,
    steps: Any | None = None,
    use_gold_context: bool = False,
) -> List[Dict[str, str]]:
    if is_socraticlm_model(teacher_model_name):
        return build_socraticlm_teacher_conversation(question, answer, steps, dialogue, action=None)
    return build_generic_response_only_teacher_conversation(question, dialogue, answer, steps, use_gold_context)


def patch_safe_acceptance_verifier(final_answer_guard: bool = True) -> Any:
    """
    Patch dice_policy_sim.verify_student_acceptance.

    Keeps old LLM verifier behavior, trims context on overflow, and only blocks a
    verifier CORRECT label when the student response clearly looks like a final
    answer but exact final-answer matching says it is wrong.
    """
    old_verify = dice_mod.verify_student_acceptance

    def safe_verify_student_acceptance(
        verifier_llm: LLM,
        question: str,
        answer: Any,
        steps: Any,
        dialogue: List[Dict[str, Any]],
        verifier_sampling_params: SamplingParams,
    ) -> Dict[str, Any]:
        last_student = dice_mod.get_last_student_response(dialogue)
        if dice_mod.is_correct(last_student, answer):
            return {
                "verifier_label": "correct",
                "verifier_text": "FINAL_ANSWER_MATCH",
                "verifier_source": "exact_answer_match",
                "verifier_false_positive_blocked": False,
            }

        trimmed_dialogue = dialogue
        while True:
            conversation = dice_mod.build_acceptance_verifier_conversation(
                question=question, answer=answer, steps=steps, dialogue=trimmed_dialogue
            )
            try:
                raw = dice_mod.generate_one_vllm(verifier_llm, conversation, verifier_sampling_params)
                label = dice_mod.parse_verifier_label(raw)
                blocked = False
                if final_answer_guard and label == "correct" and looks_like_final_answer(last_student):
                    label = "unclear"
                    blocked = True
                return {
                    "verifier_label": label,
                    "verifier_text": raw,
                    "verifier_source": "llm_verifier_final_guard" if blocked else "llm_verifier",
                    "verifier_trimmed_dialogue_len": len(trimmed_dialogue),
                    "verifier_false_positive_blocked": blocked,
                }
            except Exception as exc:
                msg = str(exc).lower()
                overflow = (
                    "maximum context length" in msg
                    or "input tokens" in msg
                    or "maximum model length" in msg
                    or "prompt (length" in msg
                    or "prompt + requested output tokens" in msg
                    or "vllmvalidationerror" in msg
                )
                if not overflow:
                    raise
                if len(trimmed_dialogue) <= 1:
                    return {
                        "verifier_label": "unclear",
                        "verifier_text": "VERIFIER_CONTEXT_OVERFLOW",
                        "verifier_source": "overflow_fallback",
                        "verifier_trimmed_dialogue_len": len(trimmed_dialogue),
                        "verifier_false_positive_blocked": False,
                    }
                trimmed_dialogue = trimmed_dialogue[2:]

    dice_mod.verify_student_acceptance = safe_verify_student_acceptance
    return old_verify


def simulate_dialogue_response_only(
    teacher_llm: LLM,
    teacher_model_name: str,
    student_model: Any,
    student_tokenizer: Any,
    student_device: torch.device,
    question: str,
    answer: Any,
    steps: Any,
    max_turns: int,
    teacher_sampling_params: SamplingParams,
    student_max_new_tokens: int,
    student_temperature: float,
    top_p: float,
    use_gold_context: bool = False,
) -> Dict[str, Any]:
    dialogue: List[Dict[str, Any]] = []
    pending_correct_turn: Optional[int] = None
    first_correct_turn: Optional[int] = None
    success = False
    end_reason = "timeout"
    any_student_correct = False
    over_intervention_count = 0
    teacher_context_trim_events = 0
    conservative_fa_count = 0
    unsafe_fa_count = 0
    step_accept_count = 0
    conservative_fa_turns: List[int] = []
    unsafe_fa_turns: List[int] = []
    step_accept_turns: List[int] = []

    cold_start_prompt = build_student_prompt_text(
        student_tokenizer=student_tokenizer,
        question=question,
        dialogue=[{"speaker": "teacher", "text": f"Can you solve this problem? {question}"}],
    )
    cold_start_text = generate_one_hf(
        model=student_model,
        tokenizer=student_tokenizer,
        prompt=cold_start_prompt,
        device=student_device,
        max_new_tokens=student_max_new_tokens,
        temperature=student_temperature,
        top_p=top_p,
    )
    cold_start_correct = is_correct(cold_start_text, answer)
    dialogue.append({"turn": 0, "speaker": "student", "text": cold_start_text, "is_correct_answer": cold_start_correct})
    if cold_start_correct:
        any_student_correct = True
        pending_correct_turn = 0
        first_correct_turn = 0

    for turn in range(1, max_turns + 1):
        trimmed_dialogue = dialogue
        while True:
            conv = build_response_only_teacher_conversation(teacher_model_name, question, trimmed_dialogue, answer, steps, use_gold_context)
            try:
                teacher_text = generate_one_vllm(teacher_llm, conv, teacher_sampling_params)
                break
            except Exception as exc:
                msg = str(exc).lower()
                overflow = any(s in msg for s in ["maximum context length", "input tokens", "maximum model length", "prompt (length", "prompt + requested output tokens", "vllmvalidationerror"])
                if not overflow:
                    raise
                if len(trimmed_dialogue) <= 1:
                    raise
                trimmed_dialogue = trimmed_dialogue[2:]
                teacher_context_trim_events += 1

        dialogue.append({"turn": turn, "speaker": "teacher", "text": teacher_text})
        last_student = [m for m in dialogue if m.get("speaker") == "student"]
        if last_student:
            diag = classify_acceptance_event(
                teacher_text=teacher_text,
                last_student_text=last_student[-1].get("text", ""),
                last_student_is_correct=bool(last_student[-1].get("is_correct_answer", False)),
            )
            dialogue[-1]["acceptance_diagnostic"] = diag
            if diag["conservative_false_acceptance"]:
                conservative_fa_count += 1
                conservative_fa_turns.append(turn)
            if diag["unsafe_final_acceptance"]:
                unsafe_fa_count += 1
                unsafe_fa_turns.append(turn)
            if diag["step_acceptance_after_nonfinal"]:
                step_accept_count += 1
                step_accept_turns.append(turn)

        if pending_correct_turn is not None:
            if teacher_accepts(teacher_text):
                success = True
                end_reason = "accepted"
                break
            over_intervention_count += 1
            pending_correct_turn = None

        student_prompt = build_student_prompt_text(student_tokenizer=student_tokenizer, question=question, dialogue=dialogue)
        student_text = generate_one_hf(
            model=student_model,
            tokenizer=student_tokenizer,
            prompt=student_prompt,
            device=student_device,
            max_new_tokens=student_max_new_tokens,
            temperature=student_temperature,
            top_p=top_p,
        )
        student_correct = is_correct(student_text, answer)
        dialogue.append({"turn": turn, "speaker": "student", "text": student_text, "is_correct_answer": student_correct})
        if student_correct:
            any_student_correct = True
            pending_correct_turn = turn
            if first_correct_turn is None:
                first_correct_turn = turn

    if not success and pending_correct_turn is not None and end_reason == "timeout":
        end_reason = "correct_but_no_teacher_confirmation"

    return {
        "success": success,
        "success_turn": pending_correct_turn if success else None,
        "first_correct_turn": first_correct_turn,
        "end_reason": end_reason,
        "over_intervention_count": over_intervention_count,
        "over_intervention": over_intervention_count > 0,
        "false_acceptance_count": conservative_fa_count,
        "false_acceptance_turns": conservative_fa_turns,
        "false_acceptance": conservative_fa_count > 0,
        "conservative_false_acceptance_count": conservative_fa_count,
        "conservative_false_acceptance_turns": conservative_fa_turns,
        "conservative_false_acceptance": conservative_fa_count > 0,
        "unsafe_final_acceptance_count": unsafe_fa_count,
        "unsafe_final_acceptance_turns": unsafe_fa_turns,
        "unsafe_final_acceptance": unsafe_fa_count > 0,
        "step_acceptance_after_nonfinal_count": step_accept_count,
        "step_acceptance_after_nonfinal_turns": step_accept_turns,
        "step_acceptance_after_nonfinal": step_accept_count > 0,
        "any_student_correct": any_student_correct,
        "teacher_context_trim_events": teacher_context_trim_events,
        "num_turns_executed": max((m["turn"] for m in dialogue), default=0),
        "intervention_count": None,
        "accept_count": None,
        "verifier_calls": 0,
        "verifier_blocks": 0,
        "verifier_false_positive": False,
        "verifier_false_positive_count": 0,
        "verifier_false_positive_blocked": False,
        "verifier_false_positive_blocked_count": 0,
        "action_trace": [],
        "dialogue": dialogue,
    }


def simulate_dialogue_dice_budgeted(
    teacher_llm: LLM,
    teacher_model_name: str,
    student_model: Any,
    student_tokenizer: Any,
    student_device: torch.device,
    policy_model: Any,
    policy_tokenizer: Any,
    policy_device: torch.device,
    question: str,
    answer: Any,
    steps: Any,
    max_turns: int,
    teacher_sampling_params: SamplingParams,
    student_max_new_tokens: int,
    student_temperature: float,
    top_p: float,
    dice_mode: str,
    intervention_cost_lambda: float,
    intervention_threshold_tau: float,
    policy_score_normalization: str,
    use_acceptance_verifier: bool,
    acceptance_override_action: str,
    verifier_sampling_params: SamplingParams,
    final_answer_guard: bool = True,
) -> Dict[str, Any]:
    old_builder = dice_mod.build_teacher_conversation
    old_verifier = patch_safe_acceptance_verifier(final_answer_guard=final_answer_guard)

    if is_socraticlm_model(teacher_model_name):
        def patched_builder(question: str, answer: Any, steps: Any, dialogue: List[Dict[str, Any]], action: str) -> List[Dict[str, str]]:
            return build_socraticlm_teacher_conversation(question, answer, steps, dialogue, action=action)
        dice_mod.build_teacher_conversation = patched_builder

    try:
        result = dice_mod.simulate_dialogue_dice(
            teacher_llm=teacher_llm,
            student_model=student_model,
            student_tokenizer=student_tokenizer,
            student_device=student_device,
            policy_model=policy_model,
            policy_tokenizer=policy_tokenizer,
            policy_device=policy_device,
            question=question,
            answer=answer,
            steps=steps,
            max_turns=max_turns,
            teacher_sampling_params=teacher_sampling_params,
            student_max_new_tokens=student_max_new_tokens,
            student_temperature=student_temperature,
            top_p=top_p,
            dice_mode=dice_mode,
            intervention_cost_lambda=intervention_cost_lambda,
            intervention_threshold_tau=intervention_threshold_tau,
            policy_score_normalization=policy_score_normalization,
            use_acceptance_verifier=use_acceptance_verifier,
            acceptance_override_action=acceptance_override_action,
            verifier_sampling_params=verifier_sampling_params,
        )
    finally:
        dice_mod.build_teacher_conversation = old_builder
        dice_mod.verify_student_acceptance = old_verifier

    conservative_fa_count = 0
    unsafe_fa_count = 0
    step_accept_count = 0
    conservative_fa_turns: List[int] = []
    unsafe_fa_turns: List[int] = []
    step_accept_turns: List[int] = []
    verifier_fp_count = 0
    verifier_fp_turns: List[int] = []
    verifier_fp_blocked_count = 0
    verifier_fp_blocked_turns: List[int] = []

    dialogue = result.get("dialogue", [])
    trace_by_turn = {int(tr.get("turn", -999)): tr for tr in result.get("action_trace", [])}

    for i, msg in enumerate(dialogue):
        if msg.get("speaker") != "teacher":
            continue
        turn = int(msg.get("turn", 0))
        prev_students = [m for m in dialogue[:i] if m.get("speaker") == "student"]
        if not prev_students:
            continue
        last_student = prev_students[-1]
        final_policy_action = msg.get("policy_action", None)
        accepted_by_policy = final_policy_action == NON_INTERVENTION_ACTION
        diag = classify_acceptance_event(
            teacher_text=msg.get("text", ""),
            last_student_text=last_student.get("text", ""),
            last_student_is_correct=bool(last_student.get("is_correct_answer", False)),
            teacher_accepted_override=accepted_by_policy,
        )
        msg["acceptance_diagnostic"] = diag

        tr = trace_by_turn.get(turn)
        verifier_info = tr.get("acceptance_verifier") if tr else None
        if verifier_info is not None:
            msg["acceptance_verifier"] = verifier_info
            blocked = bool(verifier_info.get("verifier_false_positive_blocked", False))
            msg["verifier_false_positive_blocked"] = blocked
            if blocked:
                verifier_fp_blocked_count += 1
                verifier_fp_blocked_turns.append(turn)
            if (
                accepted_by_policy
                and verifier_info.get("verifier_label") == "correct"
                and not bool(last_student.get("is_correct_answer", False))
                and looks_like_final_answer(last_student.get("text", ""))
            ):
                verifier_fp_count += 1
                verifier_fp_turns.append(turn)
                msg["verifier_false_positive"] = True
            else:
                msg["verifier_false_positive"] = False
        else:
            msg["verifier_false_positive"] = False
            msg["verifier_false_positive_blocked"] = False

        if diag["conservative_false_acceptance"]:
            conservative_fa_count += 1
            conservative_fa_turns.append(turn)
        if diag["unsafe_final_acceptance"]:
            unsafe_fa_count += 1
            unsafe_fa_turns.append(turn)
        if diag["step_acceptance_after_nonfinal"]:
            step_accept_count += 1
            step_accept_turns.append(turn)

    result["false_acceptance_count"] = conservative_fa_count
    result["false_acceptance_turns"] = conservative_fa_turns
    result["false_acceptance"] = conservative_fa_count > 0
    result["conservative_false_acceptance_count"] = conservative_fa_count
    result["conservative_false_acceptance_turns"] = conservative_fa_turns
    result["conservative_false_acceptance"] = conservative_fa_count > 0
    result["unsafe_final_acceptance_count"] = unsafe_fa_count
    result["unsafe_final_acceptance_turns"] = unsafe_fa_turns
    result["unsafe_final_acceptance"] = unsafe_fa_count > 0
    result["step_acceptance_after_nonfinal_count"] = step_accept_count
    result["step_acceptance_after_nonfinal_turns"] = step_accept_turns
    result["step_acceptance_after_nonfinal"] = step_accept_count > 0
    result["verifier_false_positive_count"] = verifier_fp_count
    result["verifier_false_positive_turns"] = verifier_fp_turns
    result["verifier_false_positive"] = verifier_fp_count > 0
    result["verifier_false_positive_blocked_count"] = verifier_fp_blocked_count
    result["verifier_false_positive_blocked_turns"] = verifier_fp_blocked_turns
    result["verifier_false_positive_blocked"] = verifier_fp_blocked_count > 0
    return result


def build_session_jobs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jobs = []
    for i, row in enumerate(rows):
        variants = []
        valid = True
        for variant in SESSION_VARIANTS:
            q = row.get(f"{variant}_question", "")
            a = row.get(f"{variant}_answer", "")
            s = row.get(f"{variant}_steps", "")
            if not q:
                valid = False
                break
            variants.append({"variant": variant, "variant_display": VARIANT_DISPLAY[variant], "question": q, "gold_answer": a, "steps": s})
        if valid:
            jobs.append({"index": i, "source_problem_id": row.get("source_problem_id"), "target_skill": row.get("target_skill"), "variants": variants})
    return jobs


def make_failed_session_record(session_job: Dict[str, Any], max_session_turns: int, error_type: str, error_message: str, tb: str) -> Dict[str, Any]:
    return {
        "index": session_job.get("index"),
        "source_problem_id": session_job.get("source_problem_id"),
        "target_skill": session_job.get("target_skill"),
        "run_type": "failed",
        "max_session_turns": max_session_turns,
        "session_success": False,
        "num_variants_reached": 0,
        "num_variants_solved": 0,
        "reached_variants": [],
        "solved_variants": [],
        "success_by_variant": {v: False for v in SESSION_VARIANTS},
        "solve_turns_by_variant": {v: None for v in SESSION_VARIANTS},
        "first_correct_turns_by_variant": {v: None for v in SESSION_VARIANTS},
        "session_end_reason": "runtime_error",
        "turns_used_total": max_session_turns,
        "budget_remaining": 0,
        "false_acceptance": False,
        "false_acceptance_count": 0,
        "conservative_false_acceptance": False,
        "conservative_false_acceptance_count": 0,
        "unsafe_final_acceptance": False,
        "unsafe_final_acceptance_count": 0,
        "step_acceptance_after_nonfinal": False,
        "step_acceptance_after_nonfinal_count": 0,
        "verifier_false_positive": False,
        "verifier_false_positive_count": 0,
        "verifier_false_positive_blocked": False,
        "verifier_false_positive_blocked_count": 0,
        "over_intervention": False,
        "over_intervention_count": 0,
        "teacher_context_trim_events": 0,
        "intervention_count": 0,
        "accept_count": 0,
        "verifier_calls": 0,
        "verifier_blocks": 0,
        "variant_results": [],
        "runtime_failed": True,
        "error_type": error_type,
        "error_message": error_message,
        "traceback": tb,
    }


def simulate_session(
    run_type: str,
    teacher_llm: LLM,
    teacher_model_name: str,
    student_model: Any,
    student_tokenizer: Any,
    student_device: torch.device,
    max_session_turns: int,
    teacher_sampling_params: SamplingParams,
    student_max_new_tokens: int,
    student_temperature: float,
    top_p: float,
    session_job: Dict[str, Any],
    policy_model: Any | None = None,
    policy_tokenizer: Any | None = None,
    policy_device: torch.device | None = None,
    dice_mode: str = "",
    intervention_cost_lambda: float = 0.0,
    intervention_threshold_tau: float = 0.5,
    policy_score_normalization: str = "mean",
    use_acceptance_verifier: bool = True,
    acceptance_override_action: str = "ask_for_justification",
    verifier_sampling_params: SamplingParams | None = None,
    response_only_gold_context: bool = False,
    final_answer_guard: bool = True,
) -> Dict[str, Any]:
    turns_used = 0
    reached_variants: List[str] = []
    solved_variants: List[str] = []
    variant_results: List[Dict[str, Any]] = []
    totals = {"cfa": 0, "ufa": 0, "step": 0, "vfp": 0, "vfpb": 0, "over": 0, "trim": 0, "interventions": 0, "accepts": 0, "vcalls": 0, "vblocks": 0}
    end_reason = "not_started"

    for variant_job in session_job["variants"]:
        remaining = max_session_turns - turns_used
        if remaining <= 0:
            end_reason = "budget_exhausted"
            break
        variant = variant_job["variant"]
        reached_variants.append(variant)

        if run_type == "response_only":
            result = simulate_dialogue_response_only(
                teacher_llm=teacher_llm,
                teacher_model_name=teacher_model_name,
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                student_device=student_device,
                question=variant_job["question"],
                answer=variant_job["gold_answer"],
                steps=variant_job["steps"],
                max_turns=remaining,
                teacher_sampling_params=teacher_sampling_params,
                student_max_new_tokens=student_max_new_tokens,
                student_temperature=student_temperature,
                top_p=top_p,
                use_gold_context=response_only_gold_context,
            )
        elif run_type == "dice":
            assert policy_model is not None and policy_tokenizer is not None and policy_device is not None and verifier_sampling_params is not None
            result = simulate_dialogue_dice_budgeted(
                teacher_llm=teacher_llm,
                teacher_model_name=teacher_model_name,
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                student_device=student_device,
                policy_model=policy_model,
                policy_tokenizer=policy_tokenizer,
                policy_device=policy_device,
                question=variant_job["question"],
                answer=variant_job["gold_answer"],
                steps=variant_job["steps"],
                max_turns=remaining,
                teacher_sampling_params=teacher_sampling_params,
                student_max_new_tokens=student_max_new_tokens,
                student_temperature=student_temperature,
                top_p=top_p,
                dice_mode=dice_mode,
                intervention_cost_lambda=intervention_cost_lambda,
                intervention_threshold_tau=intervention_threshold_tau,
                policy_score_normalization=policy_score_normalization,
                use_acceptance_verifier=use_acceptance_verifier,
                acceptance_override_action=acceptance_override_action,
                verifier_sampling_params=verifier_sampling_params,
                final_answer_guard=final_answer_guard,
            )
        else:
            raise ValueError(f"Unknown run_type={run_type}")

        local_turns = int(result.get("num_turns_executed", 0))
        cumulative_accept_turn = turns_used + local_turns
        local_first_correct = result.get("first_correct_turn", None)
        cumulative_first_correct = None if local_first_correct is None else turns_used + int(local_first_correct)
        turns_used += local_turns

        vr = {
            "variant": variant,
            "variant_display": variant_job["variant_display"],
            "question": variant_job["question"],
            "gold_answer": variant_job["gold_answer"],
            "success": result["success"],
            "local_success_turn": result["success_turn"],
            "local_first_correct_turn": result["first_correct_turn"],
            "cumulative_first_correct_turn": cumulative_first_correct,
            "cumulative_accept_turn": cumulative_accept_turn if result["success"] else None,
            "end_reason": result["end_reason"],
            "num_turns_executed": local_turns,
            "remaining_budget_after_variant": max_session_turns - turns_used,
            "over_intervention": result["over_intervention"],
            "over_intervention_count": result["over_intervention_count"],
            "false_acceptance": result.get("false_acceptance", False),
            "false_acceptance_count": result.get("false_acceptance_count", 0),
            "conservative_false_acceptance": result.get("conservative_false_acceptance", result.get("false_acceptance", False)),
            "conservative_false_acceptance_count": result.get("conservative_false_acceptance_count", result.get("false_acceptance_count", 0)),
            "unsafe_final_acceptance": result.get("unsafe_final_acceptance", False),
            "unsafe_final_acceptance_count": result.get("unsafe_final_acceptance_count", 0),
            "step_acceptance_after_nonfinal": result.get("step_acceptance_after_nonfinal", False),
            "step_acceptance_after_nonfinal_count": result.get("step_acceptance_after_nonfinal_count", 0),
            "verifier_false_positive": result.get("verifier_false_positive", False),
            "verifier_false_positive_count": result.get("verifier_false_positive_count", 0),
            "verifier_false_positive_blocked": result.get("verifier_false_positive_blocked", False),
            "verifier_false_positive_blocked_count": result.get("verifier_false_positive_blocked_count", 0),
            "any_student_correct": result["any_student_correct"],
            "intervention_count": result.get("intervention_count"),
            "accept_count": result.get("accept_count"),
            "verifier_calls": result.get("verifier_calls", 0),
            "verifier_blocks": result.get("verifier_blocks", 0),
            "action_trace": result.get("action_trace", []),
            "dialogue": result["dialogue"],
        }
        variant_results.append(vr)

        totals["cfa"] += int(vr["conservative_false_acceptance_count"])
        totals["ufa"] += int(vr["unsafe_final_acceptance_count"])
        totals["step"] += int(vr["step_acceptance_after_nonfinal_count"])
        totals["vfp"] += int(vr["verifier_false_positive_count"])
        totals["vfpb"] += int(vr["verifier_false_positive_blocked_count"])
        totals["over"] += int(result.get("over_intervention_count", 0))
        totals["trim"] += int(result.get("teacher_context_trim_events", 0))
        if result.get("intervention_count") is not None:
            totals["interventions"] += int(result.get("intervention_count", 0))
        if result.get("accept_count") is not None:
            totals["accepts"] += int(result.get("accept_count", 0))
        totals["vcalls"] += int(result.get("verifier_calls", 0))
        totals["vblocks"] += int(result.get("verifier_blocks", 0))

        if result.get("success", False):
            solved_variants.append(variant)
            end_reason = f"solved_{variant}"
        else:
            end_reason = f"failed_{variant}"
            break

    session_success = solved_variants == SESSION_VARIANTS
    solve_turns_by_variant, first_correct_turns_by_variant, success_by_variant = {}, {}, {}
    for v in SESSION_VARIANTS:
        vr = next((r for r in variant_results if r["variant"] == v), None)
        if vr is None:
            solve_turns_by_variant[v] = None
            first_correct_turns_by_variant[v] = None
            success_by_variant[v] = False
        else:
            solve_turns_by_variant[v] = vr["cumulative_accept_turn"]
            first_correct_turns_by_variant[v] = vr["cumulative_first_correct_turn"]
            success_by_variant[v] = bool(vr["success"])

    return {
        "index": session_job["index"],
        "source_problem_id": session_job["source_problem_id"],
        "target_skill": session_job["target_skill"],
        "run_type": run_type,
        "max_session_turns": max_session_turns,
        "session_success": session_success,
        "num_variants_reached": len(reached_variants),
        "num_variants_solved": len(solved_variants),
        "reached_variants": reached_variants,
        "solved_variants": solved_variants,
        "success_by_variant": success_by_variant,
        "solve_turns_by_variant": solve_turns_by_variant,
        "first_correct_turns_by_variant": first_correct_turns_by_variant,
        "session_end_reason": "solved_all" if session_success else end_reason,
        "turns_used_total": turns_used,
        "budget_remaining": max_session_turns - turns_used,
        "false_acceptance": totals["cfa"] > 0,
        "false_acceptance_count": totals["cfa"],
        "conservative_false_acceptance": totals["cfa"] > 0,
        "conservative_false_acceptance_count": totals["cfa"],
        "unsafe_final_acceptance": totals["ufa"] > 0,
        "unsafe_final_acceptance_count": totals["ufa"],
        "step_acceptance_after_nonfinal": totals["step"] > 0,
        "step_acceptance_after_nonfinal_count": totals["step"],
        "verifier_false_positive": totals["vfp"] > 0,
        "verifier_false_positive_count": totals["vfp"],
        "verifier_false_positive_blocked": totals["vfpb"] > 0,
        "verifier_false_positive_blocked_count": totals["vfpb"],
        "over_intervention": totals["over"] > 0,
        "over_intervention_count": totals["over"],
        "teacher_context_trim_events": totals["trim"],
        "intervention_count": totals["interventions"] if run_type == "dice" else None,
        "accept_count": totals["accepts"] if run_type == "dice" else None,
        "verifier_calls": totals["vcalls"],
        "verifier_blocks": totals["vblocks"],
        "variant_results": variant_results,
        "runtime_failed": False,
    }


def summarize_session_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(predictions)
    def rate(num: int, den: int = n) -> float:
        return float(num / den) if den else 0.0

    completed_easy = sum(1 for p in predictions if p["success_by_variant"].get("easy"))
    completed_source = sum(1 for p in predictions if p["success_by_variant"].get("target"))
    completed_hard = sum(1 for p in predictions if p["success_by_variant"].get("hard"))
    completed_all = sum(1 for p in predictions if p["session_success"])
    runtime_failed = sum(1 for p in predictions if p.get("runtime_failed", False))
    cfa_sessions = sum(1 for p in predictions if p.get("conservative_false_acceptance", False))
    ufa_sessions = sum(1 for p in predictions if p.get("unsafe_final_acceptance", False))
    step_sessions = sum(1 for p in predictions if p.get("step_acceptance_after_nonfinal", False))
    cfa_total = sum(int(p.get("conservative_false_acceptance_count", 0)) for p in predictions)
    ufa_total = sum(int(p.get("unsafe_final_acceptance_count", 0)) for p in predictions)
    step_total = sum(int(p.get("step_acceptance_after_nonfinal_count", 0)) for p in predictions)
    vfp_sessions = sum(1 for p in predictions if p.get("verifier_false_positive", False))
    vfp_total = sum(int(p.get("verifier_false_positive_count", 0)) for p in predictions)
    vfpb_sessions = sum(1 for p in predictions if p.get("verifier_false_positive_blocked", False))
    vfpb_total = sum(int(p.get("verifier_false_positive_blocked_count", 0)) for p in predictions)
    over_sessions = sum(1 for p in predictions if p["over_intervention"])
    over_total = sum(int(p["over_intervention_count"]) for p in predictions)
    total_turns = [int(p["turns_used_total"]) for p in predictions]
    solved_counts = [int(p["num_variants_solved"]) for p in predictions]

    solve_turn_stats, first_correct_stats = {}, {}
    for v in SESSION_VARIANTS:
        solved_turns = [p["solve_turns_by_variant"].get(v) for p in predictions if p["solve_turns_by_variant"].get(v) is not None]
        first_turns = [p["first_correct_turns_by_variant"].get(v) for p in predictions if p["first_correct_turns_by_variant"].get(v) is not None]
        solve_turn_stats[v] = {"count": len(solved_turns), "avg_cumulative_accept_turn": mean_or_zero([float(x) for x in solved_turns])}
        first_correct_stats[v] = {"count": len(first_turns), "avg_cumulative_first_correct_turn": mean_or_zero([float(x) for x in first_turns])}

    total_interventions = sum(int(p.get("intervention_count") or 0) for p in predictions)
    total_accepts = sum(int(p.get("accept_count") or 0) for p in predictions)
    verifier_calls = sum(int(p.get("verifier_calls") or 0) for p in predictions)
    verifier_blocks = sum(int(p.get("verifier_blocks") or 0) for p in predictions)
    session_composite_score = (
        rate(completed_all)
        + 0.25 * rate(completed_hard)
        + 0.10 * mean_or_zero([float(x) / 3.0 for x in solved_counts])
        - 0.05 * mean_or_zero([float(x) for x in total_turns])
        - 0.50 * rate(ufa_sessions)
        - 0.50 * rate(over_sessions)
    )
    return {
        "num_sessions": n,
        "runtime_failed_sessions": runtime_failed,
        "runtime_failure_rate": rate(runtime_failed),
        "session_successes_all_three": completed_all,
        "session_success_rate_all_three": rate(completed_all),
        "easy_successes": completed_easy,
        "easy_success_rate": rate(completed_easy),
        "source_successes": completed_source,
        "source_success_rate": rate(completed_source),
        "hard_successes": completed_hard,
        "hard_success_rate": rate(completed_hard),
        "avg_variants_solved_per_session": mean_or_zero([float(x) for x in solved_counts]),
        "avg_turns_used_per_session": mean_or_zero([float(x) for x in total_turns]),
        "avg_variants_solved_per_20_turns": mean_or_zero([float(x) / 20.0 for x in solved_counts]),
        "false_acceptance_sessions": cfa_sessions,
        "false_acceptance_rate": rate(cfa_sessions),
        "false_acceptance_count_total": cfa_total,
        "conservative_false_acceptance_sessions": cfa_sessions,
        "conservative_false_acceptance_rate": rate(cfa_sessions),
        "conservative_false_acceptance_count_total": cfa_total,
        "unsafe_final_acceptance_sessions": ufa_sessions,
        "unsafe_final_acceptance_rate": rate(ufa_sessions),
        "unsafe_final_acceptance_count_total": ufa_total,
        "step_acceptance_after_nonfinal_sessions": step_sessions,
        "step_acceptance_after_nonfinal_rate": rate(step_sessions),
        "step_acceptance_after_nonfinal_count_total": step_total,
        "verifier_false_positive_sessions": vfp_sessions,
        "verifier_false_positive_rate": rate(vfp_sessions),
        "verifier_false_positive_count_total": vfp_total,
        "verifier_false_positive_blocked_sessions": vfpb_sessions,
        "verifier_false_positive_blocked_rate": rate(vfpb_sessions),
        "verifier_false_positive_blocked_count_total": vfpb_total,
        "over_intervention_sessions": over_sessions,
        "over_intervention_rate": rate(over_sessions),
        "over_intervention_count_total": over_total,
        "total_interventions": total_interventions,
        "interventions_per_session": rate(total_interventions),
        "total_accepts": total_accepts,
        "accepts_per_session": rate(total_accepts),
        "verifier_calls": verifier_calls,
        "verifier_blocks": verifier_blocks,
        "verifier_block_rate": float(verifier_blocks / verifier_calls) if verifier_calls else 0.0,
        "solve_turn_stats": solve_turn_stats,
        "first_correct_turn_stats": first_correct_stats,
        "session_composite_score": session_composite_score,
    }


def resolve_run_type(args: argparse.Namespace) -> str:
    if args.response_only and (args.dice_map or args.dice_cost or args.dice_threshold):
        raise ValueError("Choose either --response_only or one DICE mode, not both.")
    if args.response_only:
        return "response_only"
    active = [args.dice_map, args.dice_cost, args.dice_threshold]
    if sum(bool(x) for x in active) != 1:
        raise ValueError("Choose exactly one of --response_only, --dice_map, --dice_cost, --dice_threshold.")
    return "dice"


def resolve_dice_mode(args: argparse.Namespace) -> str:
    if args.dice_map:
        return "dice_map"
    if args.dice_cost:
        return "dice_cost"
    if args.dice_threshold:
        return "dice_threshold"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Session-level tutoring evaluation over easy -> source -> hard under one shared turn budget.")
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--result_root", type=str, default=DEFAULT_RES_ROOT)
    parser.add_argument("--errors_jsonl", type=str, default="")
    parser.add_argument("--max_session_turns", type=int, default=20)
    parser.add_argument("--teacher_max_new_tokens", type=int, default=256)
    parser.add_argument("--student_max_new_tokens", type=int, default=256)
    parser.add_argument("--teacher_temperature", type=float, default=0.2)
    parser.add_argument("--student_temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL)
    parser.add_argument("--policy_base_model", type=str, default=POLICY_BASE_MODEL)
    parser.add_argument("--policy_adapter_path", type=str, default="")
    parser.add_argument("--teacher_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--teacher_pipeline_parallel_size", type=int, default=3)
    parser.add_argument("--student_device", type=int, default=3)
    parser.add_argument("--policy_device", type=int, default=3)
    parser.add_argument("--teacher_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--teacher_max_num_seqs", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--policy_use_4bit", action="store_true")
    parser.add_argument("--policy_bf16", action="store_true")
    parser.add_argument("--policy_score_normalization", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--response_only", action="store_true")
    parser.add_argument("--response_only_gold_context", action="store_true")
    parser.add_argument("--dice_map", action="store_true")
    parser.add_argument("--dice_cost", action="store_true")
    parser.add_argument("--dice_threshold", action="store_true")
    parser.add_argument("--intervention_cost_lambda", type=float, default=0.0)
    parser.add_argument("--intervention_threshold_tau", type=float, default=0.5)
    parser.add_argument("--use_acceptance_verifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--acceptance_override_action", type=str, default="ask_for_justification", choices=ACTIVE_ACTIONS)
    parser.add_argument("--verifier_max_new_tokens", type=int, default=8)
    parser.add_argument("--final_answer_guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of session jobs (triplets).",
    )
    args = parser.parse_args()

    run_type = resolve_run_type(args)
    dice_mode = resolve_dice_mode(args)
    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 0:
        raise RuntimeError("At least one visible GPU is required.")
    teacher_world_size = args.teacher_tensor_parallel_size * args.teacher_pipeline_parallel_size
    if teacher_world_size >= visible_gpus:
        raise ValueError("Teacher TP*PP must be < visible GPUs so one GPU remains for student/policy.")
    if args.student_device < 0 or args.student_device >= visible_gpus:
        raise ValueError(f"--student_device={args.student_device} out of range for visible GPUs={visible_gpus}")
    if args.student_device < teacher_world_size:
        raise ValueError(f"student_device={args.student_device} overlaps with teacher GPU slice [0..{teacher_world_size - 1}].")
    if run_type == "dice":
        if not args.policy_adapter_path:
            raise ValueError("--policy_adapter_path is required for DICE session evaluation.")
        if args.policy_device < 0 or args.policy_device >= visible_gpus:
            raise ValueError(f"--policy_device={args.policy_device} out of range for visible GPUs={visible_gpus}")
        if args.policy_device < teacher_world_size:
            raise ValueError(f"policy_device={args.policy_device} overlaps with teacher GPU slice [0..{teacher_world_size - 1}].")

    cfg = AutoConfig.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher_heads = int(getattr(cfg, "num_attention_heads", 0))
    if teacher_heads <= 0:
        raise ValueError("Could not infer teacher model num_attention_heads from config.")
    if teacher_heads % args.teacher_tensor_parallel_size != 0:
        raise ValueError(f"Teacher TP={args.teacher_tensor_parallel_size} is invalid for attention_heads={teacher_heads}.")

    rows = load_json(Path(args.input_json))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        rows = rows[: args.limit]
        print(f"Using first {len(rows)} triplets (--limit={args.limit}).")
    jobs = build_session_jobs(rows)
    print(f"Loaded {len(rows)} rows; built {len(jobs)} session jobs.")
    print(f"Run type: {run_type}")
    print(f"Teacher model: {args.teacher_model}")
    if run_type == "dice":
        print(f"DICE mode: {dice_mode}")

    print("Loading student model...")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    student_device = torch.device(f"cuda:{args.student_device}")
    student_torch_dtype = torch.float16 if args.dtype in {"float16", "half"} else torch.bfloat16
    student_model = AutoModelForCausalLM.from_pretrained(args.student_model, trust_remote_code=True, torch_dtype=student_torch_dtype)
    student_model.to(student_device)
    student_model.eval()

    policy_model = None
    policy_tokenizer = None
    policy_device = None
    if run_type == "dice":
        print("Loading policy model...")
        policy_device = torch.device(f"cuda:{args.policy_device}")
        policy_model, policy_tokenizer = load_policy_model_and_tokenizer(
            base_model_name=args.policy_base_model,
            adapter_path=args.policy_adapter_path,
            device=policy_device,
            use_4bit=args.policy_use_4bit,
            bf16=args.policy_bf16,
        )

    print("Loading teacher generator with vLLM...")
    teacher_llm = build_llm(
        model_name=args.teacher_model,
        tensor_parallel_size=args.teacher_tensor_parallel_size,
        pipeline_parallel_size=args.teacher_pipeline_parallel_size,
        gpu_memory_utilization=args.teacher_gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.teacher_max_num_seqs,
        dtype=args.dtype,
    )
    teacher_sampling_params = SamplingParams(temperature=args.teacher_temperature, top_p=args.top_p, max_tokens=args.teacher_max_new_tokens)
    verifier_sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.verifier_max_new_tokens)

    teacher_slug = model_slug(args.teacher_model)
    student_slug = model_slug(args.student_model)
    if run_type == "response_only":
        mode_tag = "response_only"
        if args.response_only_gold_context:
            mode_tag += "_goldctx"
        combo_slug = f"{mode_tag}__gen_{teacher_slug}__student_{student_slug}"
    else:
        policy_slug = model_slug(args.policy_base_model)
        adapter_slug = Path(args.policy_adapter_path).name
        if dice_mode == "dice_cost":
            mode_tag = f"dice_cost_lam{args.intervention_cost_lambda:g}"
        elif dice_mode == "dice_threshold":
            mode_tag = f"dice_threshold_tau{args.intervention_threshold_tau:g}"
        else:
            mode_tag = "dice_map"
        verifier_tag = "verifier_on" if args.use_acceptance_verifier else "verifier_off"
        guard_tag = "finalguard_on" if args.final_answer_guard else "finalguard_off"
        mode_tag = f"{mode_tag}_{verifier_tag}_{guard_tag}"
        combo_slug = f"{mode_tag}__policy_{policy_slug}_{adapter_slug}__gen_{teacher_slug}__student_{student_slug}"

    pred_dir = Path(args.output_root) / combo_slug
    res_dir = Path(args.result_root) / combo_slug
    pred_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "preds_session_easy_source_hard.jsonl"
    res_path = res_dir / "metrics_session_easy_source_hard.json"
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else pred_dir / "errors_session_easy_source_hard.jsonl"
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: List[Dict[str, Any]] = []
    failed_jobs = 0
    for job in tqdm(jobs, desc=f"Session eval {combo_slug}", unit="session"):
        try:
            session = simulate_session(
                run_type=run_type,
                teacher_llm=teacher_llm,
                teacher_model_name=args.teacher_model,
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                student_device=student_device,
                max_session_turns=args.max_session_turns,
                teacher_sampling_params=teacher_sampling_params,
                student_max_new_tokens=args.student_max_new_tokens,
                student_temperature=args.student_temperature,
                top_p=args.top_p,
                session_job=job,
                policy_model=policy_model,
                policy_tokenizer=policy_tokenizer,
                policy_device=policy_device,
                dice_mode=dice_mode,
                intervention_cost_lambda=args.intervention_cost_lambda,
                intervention_threshold_tau=args.intervention_threshold_tau,
                policy_score_normalization=args.policy_score_normalization,
                use_acceptance_verifier=args.use_acceptance_verifier,
                acceptance_override_action=args.acceptance_override_action,
                verifier_sampling_params=verifier_sampling_params,
                response_only_gold_context=args.response_only_gold_context,
                final_answer_guard=args.final_answer_guard,
            )
        except Exception as exc:
            failed_jobs += 1
            tb = traceback.format_exc()
            session = make_failed_session_record(job, args.max_session_turns, type(exc).__name__, str(exc), tb)
            append_jsonl(errors_path, {
                "index": job.get("index"),
                "source_problem_id": job.get("source_problem_id"),
                "target_skill": job.get("target_skill"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": tb,
            })
        predictions.append(session)
        append_jsonl(pred_path, session)

    summary = summarize_session_predictions(predictions)
    metrics = {
        "run_type": run_type,
        "dice_mode": dice_mode if run_type == "dice" else None,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "policy_base_model": args.policy_base_model if run_type == "dice" else None,
        "policy_adapter_path": args.policy_adapter_path if run_type == "dice" else None,
        "input_json": args.input_json,
        "max_session_turns": args.max_session_turns,
        "session_variants": SESSION_VARIANTS,
        "num_session_jobs_total": len(jobs),
        "num_session_jobs_failed": failed_jobs,
        **summary,
        "predictions_path": str(pred_path),
        "errors_path": str(errors_path),
        "config": {
            "teacher_max_new_tokens": args.teacher_max_new_tokens,
            "student_max_new_tokens": args.student_max_new_tokens,
            "teacher_temperature": args.teacher_temperature,
            "student_temperature": args.student_temperature,
            "top_p": args.top_p,
            "visible_gpus": visible_gpus,
            "teacher_tensor_parallel_size": args.teacher_tensor_parallel_size,
            "teacher_pipeline_parallel_size": args.teacher_pipeline_parallel_size,
            "student_device": args.student_device,
            "policy_device": args.policy_device if run_type == "dice" else None,
            "teacher_gpu_memory_utilization": args.teacher_gpu_memory_utilization,
            "teacher_max_num_seqs": args.teacher_max_num_seqs,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "policy_use_4bit": args.policy_use_4bit if run_type == "dice" else None,
            "policy_bf16": args.policy_bf16 if run_type == "dice" else None,
            "policy_score_normalization": args.policy_score_normalization if run_type == "dice" else None,
            "intervention_cost_lambda": args.intervention_cost_lambda if dice_mode == "dice_cost" else None,
            "intervention_threshold_tau": args.intervention_threshold_tau if dice_mode == "dice_threshold" else None,
            "use_acceptance_verifier": args.use_acceptance_verifier if run_type == "dice" else None,
            "acceptance_override_action": args.acceptance_override_action if run_type == "dice" else None,
            "verifier_max_new_tokens": args.verifier_max_new_tokens if run_type == "dice" else None,
            "final_answer_guard": args.final_answer_guard if run_type == "dice" else None,
            "response_only_gold_context": args.response_only_gold_context if run_type == "response_only" else None,
        },
    }
    with res_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved predictions to: {pred_path}")
    print(f"Saved metrics to:     {res_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

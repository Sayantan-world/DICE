from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from vllm import LLM, SamplingParams


TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
POLICY_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DEFAULT_INPUT = "data/triplets_250.json"
DEFAULT_PRED_ROOT = "outputs/preds"
DEFAULT_RES_ROOT = "outputs/metrics"

ACTION_SET = [
    "accept_and_advance",
    "arithmetic_repair",
    "ask_for_justification",
    "conceptual_hint",
    "handle_question",
    "refocus_subgoal",
]
NON_INTERVENTION_ACTION = "accept_and_advance"
ACTIVE_ACTIONS = [a for a in ACTION_SET if a != NON_INTERVENTION_ACTION]

ACTION_DESCRIPTIONS = {
    "accept_and_advance": (
        "The student is sufficiently on track. Briefly acknowledge and let the student continue."
    ),
    "arithmetic_repair": (
        "The student made an arithmetic or numerical computation mistake. Guide them to recompute."
    ),
    "ask_for_justification": (
        "The student may be correct or partially correct but needs to justify their reasoning."
    ),
    "conceptual_hint": (
        "The student has a conceptual misunderstanding. Give a hint about the underlying idea."
    ),
    "handle_question": (
        "The student asked a question. Answer or clarify it without revealing the full solution."
    ),
    "refocus_subgoal": (
        "The student skipped, jumped ahead, or drifted from the current subgoal. Refocus them."
    ),
}

ACCEPT_PATTERNS = [
    r"(?i)\b(correct|exactly|that'?s right|well done|great job|excellent|perfect|you got it)\b",
    r"(?i)\b(yes|right)[!.,]",
    r"(?i)now.{0,20}(next|move on|let'?s continue)",
    r"(?i)\b(confirm|indeed|absolutely|precisely)\b",
    r"(?i)\bcorrect\w*[!.]",
    r"(?i)you'?ve.{0,30}(done|calculated|solved|understood|got)",
    r"(?i)keep up the good work",
]


# ----------------------------
# Utilities
# ----------------------------

def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of datapoints.")
    return data


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def mean_or_zero(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def resolve_variants(args: argparse.Namespace) -> List[str]:
    requested: List[str] = []
    if args.target:
        requested.append("target")
    if args.easy:
        requested.append("easy")
    if args.hard:
        requested.append("hard")
    return requested or ["target"]


def resolve_dice_mode(args: argparse.Namespace) -> str:
    modes = [
        ("dice_map", args.dice_map),
        ("dice_cost", args.dice_cost),
        ("dice_threshold", args.dice_threshold),
    ]
    active = [name for name, enabled in modes if enabled]
    if len(active) != 1:
        raise ValueError(
            "Choose exactly one inference mode: --dice_map, --dice_cost, or --dice_threshold. "
            "This script is for DICE policy-conditioned simulation. For plain SocraLM/LM evaluation, "
            "use the original SocraLM script."
        )
    return active[0]


def normalize_answer(text: str) -> Tuple[str, float | None]:
    s = str(text).strip()
    s = s.replace("$", "")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;!?\"'`")
    s_nocomma = s.replace(",", "")
    try:
        return s.lower(), float(s_nocomma)
    except ValueError:
        nums = re.findall(r"[-+]?\d*\.?\d+", s_nocomma)
        if nums:
            try:
                return s.lower(), float(nums[-1])
            except ValueError:
                pass
        return s.lower(), None


def is_correct(pred: str, gold: Any) -> bool:
    pred_norm, pred_num = normalize_answer(pred)
    gold_norm, gold_num = normalize_answer(str(gold))
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6
    return pred_norm == gold_norm


def teacher_accepts(teacher_utterance: str) -> bool:
    for pattern in ACCEPT_PATTERNS:
        if re.search(pattern, teacher_utterance):
            return True
    return False


# ----------------------------
# Student prompts
# ----------------------------

def build_student_system_prompt(question: str) -> str:
    return (
        "You are a student solving a math problem with a teacher.\n"
        "Think internally before answering, but never reveal internal reasoning.\n"
        "Answer in about 50 words and explain briefly why.\n"
        "If you are confident, provide a final numeric answer.\n"
        f"\n[Problem]\n{question}\n"
    )


def build_student_prompt_text(
    student_tokenizer: Any,
    question: str,
    dialogue: List[Dict[str, Any]],
) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_student_system_prompt(question)}
    ]
    for msg in dialogue:
        role = "user" if msg["speaker"] == "teacher" else "assistant"
        messages.append({"role": role, "content": msg["text"]})
    # Some tokenizers (e.g., certain Llama checkpoints) do not ship a chat template.
    # Fall back to a plain role-tagged prompt so runs don't crash on model swaps.
    try:
        return student_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "chat template" not in msg:
            raise

    lines: List[str] = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


# ----------------------------
# Shared dialogue helpers
# ----------------------------

def format_dialogue_for_policy(dialogue: List[Dict[str, Any]]) -> str:
    if not dialogue:
        return "None"
    lines = []
    for msg in dialogue:
        speaker = msg.get("speaker", "")
        text = msg.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def get_last_student_response(dialogue: List[Dict[str, Any]]) -> str:
    students = [m for m in dialogue if m.get("speaker") == "student"]
    return students[-1]["text"] if students else ""


def get_last_teacher_utterance(dialogue: List[Dict[str, Any]]) -> str:
    teachers = [m for m in dialogue if m.get("speaker") == "teacher"]
    return teachers[-1]["text"] if teachers else "Can you solve this problem?"


# ----------------------------
# Policy prompts
# ----------------------------

def build_policy_messages(
    question: str,
    answer: Any,
    steps: Any,
    dialogue: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    history = format_dialogue_for_policy(dialogue)
    last_teacher = get_last_teacher_utterance(dialogue)
    last_student = get_last_student_response(dialogue)

    action_list = "\n".join(f"- {a}: {ACTION_DESCRIPTIONS[a]}" for a in ACTION_SET)

    system = (
        "You are a tutoring policy model. Your task is to choose exactly one pedagogical action.\n"
        "Return only the action label and nothing else.\n"
    )

    user = (
        "[Problem]\n"
        f"{question}\n\n"
        "[Gold Answer]\n"
        f"{answer}\n\n"
        "[Solution Steps]\n"
        f"{steps}\n\n"
        "[Dialogue History]\n"
        f"{history}\n\n"
        "[Current Tutor Prompt]\n"
        f"{last_teacher}\n\n"
        "[Latest Student Response]\n"
        f"{last_student}\n\n"
        "[Allowed Actions]\n"
        f"{action_list}\n\n"
        "Choose the best action label from the allowed actions. "
        "Return only one label."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_policy_prompt(policy_tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    return policy_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ----------------------------
# Teacher generation prompts
# ----------------------------

def build_teacher_system_prompt(question: str, answer: Any, steps: Any, action: str) -> str:
    return (
        "You are a Socratic math tutor. Your goal is to help the student solve the problem "
        "without directly revealing the final answer unless it is already clearly solved.\n"
        f"The selected pedagogical action is: {action}.\n"
        f"Action meaning: {ACTION_DESCRIPTIONS[action]}\n"
        "Generate one concise teacher utterance that follows this action.\n"
    )


def build_teacher_conversation(
    question: str,
    answer: Any,
    steps: Any,
    dialogue: List[Dict[str, Any]],
    action: str,
) -> List[Dict[str, str]]:
    user_seed = (
        f"[Problem] {question}\n"
        f"[Answer] {answer}\n"
        f"[Analysis / Steps] {steps}\n"
        f"[Selected Action] {action}\n"
        f"[Action Meaning] {ACTION_DESCRIPTIONS[action]}\n"
    )

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": build_teacher_system_prompt(question, answer, steps, action),
        },
        {"role": "user", "content": user_seed},
    ]

    if not dialogue:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Please start tutoring me. Ask one concise Socratic question first. "
                    "Do not reveal the final answer immediately."
                ),
            }
        )
    else:
        for msg in dialogue:
            role = "assistant" if msg["speaker"] == "teacher" else "user"
            messages.append({"role": role, "content": msg["text"]})

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Use the selected action {action}. "
                    "Generate the next teacher utterance in a Socratic style. "
                    "Be concise and do not reveal unnecessary solution steps."
                ),
            }
        )

    return messages


def deterministic_accept_utterance(verified_correct: bool = False) -> str:
    if verified_correct:
        return "That's correct. Please continue to the next step."
    return "Okay, continue with your next step."


# ----------------------------
# Model loading and generation
# ----------------------------

def build_llm(
    model_name: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    dtype: str,
) -> LLM:
    return LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        dtype=dtype,
        enable_chunked_prefill=True,
    )


def generate_one_vllm(
    llm: LLM,
    conversation: List[Dict[str, str]],
    sampling_params: SamplingParams,
) -> str:
    out = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)[0]
    return out.outputs[0].text.strip()


def generate_one_hf(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    do_sample = temperature > 0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **gen_kwargs)

    prompt_len = encoded["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def is_peft_adapter(path: str) -> bool:
    local = Path(path)
    if local.exists():
        return (local / "adapter_config.json").is_file()
    try:
        from huggingface_hub import file_exists

        return bool(file_exists(repo_id=path, filename="adapter_config.json"))
    except Exception:
        return False


def load_policy_model_and_tokenizer(
    base_model_name: str,
    adapter_path: str,
    device: torch.device,
    use_4bit: bool,
    bf16: bool,
):
    load_as_adapter = is_peft_adapter(adapter_path)
    tokenizer_src = base_model_name if load_as_adapter else adapter_path
    model_src = base_model_name if load_as_adapter else adapter_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        trust_remote_code=True,
        use_fast=True,
    )
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
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        quantization_config=quant_config,
        device_map={"": device.index} if device.type == "cuda" else None,
    )
    if load_as_adapter:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


# ----------------------------
# Acceptance verifier
# ----------------------------

def parse_verifier_label(text: str) -> str:
    t = (text or "").strip().upper()

    # Order matters: "INCORRECT" contains "CORRECT".
    if "INCORRECT" in t:
        return "incorrect"
    if "UNCLEAR" in t or "UNSURE" in t or "UNKNOWN" in t:
        return "unclear"
    if "CORRECT" in t:
        return "correct"

    return "unclear"


def build_acceptance_verifier_conversation(
    question: str,
    answer: Any,
    steps: Any,
    dialogue: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    history = format_dialogue_for_policy(dialogue)
    last_teacher = get_last_teacher_utterance(dialogue)
    last_student = get_last_student_response(dialogue)

    system = (
        "You are a strict verifier for a math tutoring dialogue.\n"
        "Your job is to decide whether the latest student response is safe to accept.\n"
        "Return exactly one label: CORRECT, INCORRECT, or UNCLEAR.\n\n"
        "Use CORRECT only if the student's latest response is mathematically valid for the current step "
        "or gives the correct final answer.\n"
        "Use INCORRECT if it contains a clear arithmetic, conceptual, or trajectory error.\n"
        "Use UNCLEAR if the response is incomplete, ambiguous, unsupported, or cannot be safely accepted.\n"
        "Do not explain."
    )

    user = (
        "[Problem]\n"
        f"{question}\n\n"
        "[Gold Answer]\n"
        f"{answer}\n\n"
        "[Reference Solution Steps]\n"
        f"{steps}\n\n"
        "[Dialogue History]\n"
        f"{history}\n\n"
        "[Current Tutor Prompt]\n"
        f"{last_teacher}\n\n"
        "[Latest Student Response]\n"
        f"{last_student}\n\n"
        "Can the tutor safely accept the latest student response and let the student continue?\n"
        "Return exactly one label: CORRECT, INCORRECT, or UNCLEAR."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def verify_student_acceptance(
    verifier_llm: LLM,
    question: str,
    answer: Any,
    steps: Any,
    dialogue: List[Dict[str, Any]],
    verifier_sampling_params: SamplingParams,
) -> Dict[str, Any]:
    last_student = get_last_student_response(dialogue)

    # Fast deterministic shortcut: if the latest student response contains the exact final answer,
    # it is safe to accept as final-answer correct.
    if is_correct(last_student, answer):
        return {
            "verifier_label": "correct",
            "verifier_text": "FINAL_ANSWER_MATCH",
            "verifier_source": "exact_answer_match",
        }

    conversation = build_acceptance_verifier_conversation(
        question=question,
        answer=answer,
        steps=steps,
        dialogue=dialogue,
    )
    raw = generate_one_vllm(verifier_llm, conversation, verifier_sampling_params)
    label = parse_verifier_label(raw)

    return {
        "verifier_label": label,
        "verifier_text": raw,
        "verifier_source": "llm_verifier",
    }


# ----------------------------
# Policy scoring and DICE inference
# ----------------------------

@torch.no_grad()
def score_action_labels(
    policy_model: Any,
    policy_tokenizer: Any,
    policy_prompt_text: str,
    device: torch.device,
    score_normalization: str = "mean",
) -> Dict[str, float]:
    """
    Computes constrained label scores for each action label.

    Score is the log-likelihood of the action label after the prompt.
    If score_normalization='mean', score is normalized by label token length.
    If score_normalization='sum', score is total label log-likelihood.
    """
    encoded_prompt = policy_tokenizer(policy_prompt_text, return_tensors="pt")
    input_ids_prompt = encoded_prompt["input_ids"].to(device)
    attention_prompt = encoded_prompt["attention_mask"].to(device)

    scores: Dict[str, float] = {}

    for action in ACTION_SET:
        label_text = action
        label_ids = policy_tokenizer(
            label_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)

        full_input_ids = torch.cat([input_ids_prompt, label_ids], dim=1)
        full_attention = torch.cat(
            [attention_prompt, torch.ones_like(label_ids, device=device)],
            dim=1,
        )

        outputs = policy_model(
            input_ids=full_input_ids,
            attention_mask=full_attention,
        )

        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        prompt_len = input_ids_prompt.shape[1]
        label_len = label_ids.shape[1]

        token_logps = []
        for j in range(label_len):
            pred_pos = prompt_len + j - 1
            token_id = label_ids[0, j]
            token_logps.append(log_probs[0, pred_pos, token_id])

        total = torch.stack(token_logps).sum()
        if score_normalization == "mean":
            score = total / max(label_len, 1)
        elif score_normalization == "sum":
            score = total
        else:
            raise ValueError("--policy_score_normalization must be 'mean' or 'sum'.")

        scores[action] = float(score.detach().cpu().item())

    return scores


def scores_to_log_probs(action_scores: Dict[str, float]) -> Dict[str, float]:
    score_tensor = torch.tensor([action_scores[a] for a in ACTION_SET], dtype=torch.float32)
    logp_tensor = F.log_softmax(score_tensor, dim=-1)
    return {a: float(logp_tensor[i].item()) for i, a in enumerate(ACTION_SET)}


def choose_action_from_policy(
    action_log_probs: Dict[str, float],
    mode: str,
    intervention_cost_lambda: float,
    intervention_threshold_tau: float,
) -> Tuple[str, Dict[str, Any]]:
    probs = {a: math.exp(action_log_probs[a]) for a in ACTION_SET}
    p_accept = probs[NON_INTERVENTION_ACTION]
    p_intervene = 1.0 - p_accept

    if mode == "dice_map":
        chosen = max(ACTION_SET, key=lambda a: action_log_probs[a])
        decision_scores = {a: action_log_probs[a] for a in ACTION_SET}

    elif mode == "dice_cost":
        decision_scores = {
            a: action_log_probs[a] - (
                intervention_cost_lambda if a != NON_INTERVENTION_ACTION else 0.0
            )
            for a in ACTION_SET
        }
        chosen = max(ACTION_SET, key=lambda a: decision_scores[a])

    elif mode == "dice_threshold":
        if p_intervene <= intervention_threshold_tau:
            chosen = NON_INTERVENTION_ACTION
        else:
            chosen = max(ACTIVE_ACTIONS, key=lambda a: action_log_probs[a])
        decision_scores = {a: action_log_probs[a] for a in ACTION_SET}

    else:
        raise ValueError(f"Unknown DICE mode: {mode}")

    best_active = max(ACTIVE_ACTIONS, key=lambda a: action_log_probs[a])
    active_margin = (
        action_log_probs[best_active]
        - action_log_probs[NON_INTERVENTION_ACTION]
    )

    info = {
        "mode": mode,
        "chosen_action": chosen,
        "action_log_probs": action_log_probs,
        "action_probs": probs,
        "decision_scores": decision_scores,
        "p_accept": p_accept,
        "p_intervene": p_intervene,
        "best_active_action": best_active,
        "best_active_logprob_margin_over_accept": active_margin,
        "intervention_cost_lambda": intervention_cost_lambda,
        "intervention_threshold_tau": intervention_threshold_tau,
    }
    return chosen, info


def infer_policy_action(
    policy_model: Any,
    policy_tokenizer: Any,
    policy_device: torch.device,
    question: str,
    answer: Any,
    steps: Any,
    dialogue: List[Dict[str, Any]],
    mode: str,
    intervention_cost_lambda: float,
    intervention_threshold_tau: float,
    score_normalization: str,
) -> Tuple[str, Dict[str, Any]]:
    messages = build_policy_messages(
        question=question,
        answer=answer,
        steps=steps,
        dialogue=dialogue,
    )
    prompt_text = render_policy_prompt(policy_tokenizer, messages)

    action_scores = score_action_labels(
        policy_model=policy_model,
        policy_tokenizer=policy_tokenizer,
        policy_prompt_text=prompt_text,
        device=policy_device,
        score_normalization=score_normalization,
    )
    action_log_probs = scores_to_log_probs(action_scores)

    action, info = choose_action_from_policy(
        action_log_probs=action_log_probs,
        mode=mode,
        intervention_cost_lambda=intervention_cost_lambda,
        intervention_threshold_tau=intervention_threshold_tau,
    )
    info["raw_action_scores"] = action_scores
    return action, info


# ----------------------------
# Simulation
# ----------------------------

def simulate_dialogue_dice(
    teacher_llm: LLM,
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
) -> Dict[str, Any]:
    dialogue: List[Dict[str, Any]] = []
    pending_correct_turn: int | None = None
    first_correct_turn: int | None = None
    success = False
    end_reason = "timeout"

    over_intervention_count = 0
    any_student_correct = False
    teacher_context_trim_events = 0

    false_acceptance_count = 0
    false_acceptance_turns: List[int] = []

    action_trace: List[Dict[str, Any]] = []
    intervention_count = 0
    accept_count = 0

    verifier_calls = 0
    verifier_blocks = 0
    verifier_correct = 0
    verifier_incorrect = 0
    verifier_unclear = 0

    # Turn 0: student attempts the problem before tutoring starts.
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

    dialogue.append(
        {
            "turn": 0,
            "speaker": "student",
            "text": cold_start_text,
            "is_correct_answer": cold_start_correct,
        }
    )

    if cold_start_correct:
        any_student_correct = True
        pending_correct_turn = 0
        first_correct_turn = 0

    for turn in range(1, max_turns + 1):
        selected_action, policy_info = infer_policy_action(
            policy_model=policy_model,
            policy_tokenizer=policy_tokenizer,
            policy_device=policy_device,
            question=question,
            answer=answer,
            steps=steps,
            dialogue=dialogue,
            mode=dice_mode,
            intervention_cost_lambda=intervention_cost_lambda,
            intervention_threshold_tau=intervention_threshold_tau,
            score_normalization=policy_score_normalization,
        )

        original_selected_action = selected_action
        verifier_info: Optional[Dict[str, Any]] = None

        # Verifier gate: do not allow accept_and_advance unless latest student response is safe to accept.
        if selected_action == NON_INTERVENTION_ACTION and use_acceptance_verifier:
            verifier_info = verify_student_acceptance(
                verifier_llm=teacher_llm,
                question=question,
                answer=answer,
                steps=steps,
                dialogue=dialogue,
                verifier_sampling_params=verifier_sampling_params,
            )
            verifier_calls += 1

            verifier_label = verifier_info["verifier_label"]
            if verifier_label == "correct":
                verifier_correct += 1
            elif verifier_label == "incorrect":
                verifier_incorrect += 1
            else:
                verifier_unclear += 1

            if verifier_label != "correct":
                verifier_blocks += 1
                selected_action = acceptance_override_action
                policy_info["verifier_overrode_accept"] = True
            else:
                policy_info["verifier_overrode_accept"] = False

        if selected_action == NON_INTERVENTION_ACTION:
            teacher_text = deterministic_accept_utterance(verified_correct=True)
            accept_count += 1
        else:
            intervention_count += 1
            trimmed_dialogue = dialogue
            while True:
                teacher_conversation = build_teacher_conversation(
                    question=question,
                    answer=answer,
                    steps=steps,
                    dialogue=trimmed_dialogue,
                    action=selected_action,
                )
                try:
                    teacher_text = generate_one_vllm(
                        teacher_llm,
                        teacher_conversation,
                        teacher_sampling_params,
                    )
                    break
                except Exception as exc:
                    msg = str(exc)
                    overflow = (
                        "maximum context length" in msg.lower()
                        or "input tokens" in msg.lower()
                        or "maximum model length" in msg.lower()
                        or "prompt (length" in msg.lower()
                        or "prompt + requested output tokens" in msg.lower()
                        or "vllmvalidationerror" in msg.lower()
                    )
                    if not overflow:
                        raise
                    if len(trimmed_dialogue) <= 1:
                        raise
                    trimmed_dialogue = trimmed_dialogue[2:]
                    teacher_context_trim_events += 1

        dialogue.append(
            {
                "turn": turn,
                "speaker": "teacher",
                "text": teacher_text,
                "policy_action": selected_action,
                "original_policy_action": original_selected_action,
            }
        )

        policy_info_record = dict(policy_info)
        policy_info_record["turn"] = turn
        policy_info_record["teacher_text"] = teacher_text
        policy_info_record["original_selected_action"] = original_selected_action
        policy_info_record["final_selected_action"] = selected_action
        policy_info_record["acceptance_verifier"] = verifier_info
        action_trace.append(policy_info_record)

        # False acceptance: tutor accepts even though latest student answer is not correct.
        # With verifier on, this should be much lower. This still uses the existing final-answer correctness
        # detector, so it is conservative for intermediate steps.
        if teacher_accepts(teacher_text):
            last_student = [m for m in dialogue if m.get("speaker") == "student"]
            if last_student and not last_student[-1].get("is_correct_answer", False):
                false_acceptance_count += 1
                false_acceptance_turns.append(turn)

        # If the student was already correct, the teacher should accept. If it intervenes, count over-intervention.
        if pending_correct_turn is not None:
            if teacher_accepts(teacher_text):
                success = True
                end_reason = "accepted"
                break
            over_intervention_count += 1
            pending_correct_turn = None

        student_prompt = build_student_prompt_text(
            student_tokenizer=student_tokenizer,
            question=question,
            dialogue=dialogue,
        )
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

        dialogue.append(
            {
                "turn": turn,
                "speaker": "student",
                "text": student_text,
                "is_correct_answer": student_correct,
            }
        )

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
        "false_acceptance_count": false_acceptance_count,
        "false_acceptance_turns": false_acceptance_turns,
        "false_acceptance": false_acceptance_count > 0,
        "any_student_correct": any_student_correct,
        "teacher_context_trim_events": teacher_context_trim_events,
        "num_turns_executed": max((m["turn"] for m in dialogue), default=0),
        "intervention_count": intervention_count,
        "accept_count": accept_count,
        "verifier_calls": verifier_calls,
        "verifier_blocks": verifier_blocks,
        "verifier_correct": verifier_correct,
        "verifier_incorrect": verifier_incorrect,
        "verifier_unclear": verifier_unclear,
        "action_trace": action_trace,
        "dialogue": dialogue,
    }


# ----------------------------
# Metrics
# ----------------------------

def summarize_predictions(
    predictions: List[Dict[str, Any]],
    variants: List[str],
) -> Dict[str, Any]:
    total = len(predictions)
    successes = sum(1 for p in predictions if p["success"])
    with_any_correct = [p for p in predictions if p["any_student_correct"]]

    over_int_sessions = sum(1 for p in predictions if p["over_intervention"])
    over_int_given_correct = sum(1 for p in with_any_correct if p["over_intervention"])

    false_accept_sessions = sum(1 for p in predictions if p.get("false_acceptance", False))
    false_accept_count_total = sum(int(p.get("false_acceptance_count", 0)) for p in predictions)

    teacher_context_trim_events_total = sum(int(p.get("teacher_context_trim_events", 0)) for p in predictions)

    total_interventions = sum(int(p.get("intervention_count", 0)) for p in predictions)
    total_accepts = sum(int(p.get("accept_count", 0)) for p in predictions)
    total_teacher_turns = total_interventions + total_accepts

    turn_counts = [int(p.get("num_turns_executed", 0)) for p in predictions]
    success_turn_counts = [int(p.get("num_turns_executed", 0)) for p in predictions if p.get("success")]
    failure_turn_counts = [int(p.get("num_turns_executed", 0)) for p in predictions if not p.get("success")]

    verifier_calls_total = sum(int(p.get("verifier_calls", 0)) for p in predictions)
    verifier_blocks_total = sum(int(p.get("verifier_blocks", 0)) for p in predictions)
    verifier_correct_total = sum(int(p.get("verifier_correct", 0)) for p in predictions)
    verifier_incorrect_total = sum(int(p.get("verifier_incorrect", 0)) for p in predictions)
    verifier_unclear_total = sum(int(p.get("verifier_unclear", 0)) for p in predictions)

    action_counter = {a: 0 for a in ACTION_SET}
    final_action_counter = {a: 0 for a in ACTION_SET}
    verifier_override_counter = {a: 0 for a in ACTION_SET}

    for p in predictions:
        for item in p.get("action_trace", []):
            a = item.get("chosen_action")
            if a in action_counter:
                action_counter[a] += 1

            fa = item.get("final_selected_action", a)
            if fa in final_action_counter:
                final_action_counter[fa] += 1

            if item.get("original_selected_action") == NON_INTERVENTION_ACTION and fa != NON_INTERVENTION_ACTION:
                if fa in verifier_override_counter:
                    verifier_override_counter[fa] += 1

    by_variant: Dict[str, Dict[str, Any]] = {}
    for v in variants:
        sub = [p for p in predictions if p["variant"] == v]
        n = len(sub)
        s = sum(1 for p in sub if p["success"])
        oi = sum(1 for p in sub if p["over_intervention"])
        fa_sess = sum(1 for p in sub if p.get("false_acceptance", False))
        fa_cnt = sum(int(p.get("false_acceptance_count", 0)) for p in sub)
        sub_with_correct = [p for p in sub if p["any_student_correct"]]
        oi_gc = sum(1 for p in sub_with_correct if p["over_intervention"])

        interventions = sum(int(p.get("intervention_count", 0)) for p in sub)
        accepts = sum(int(p.get("accept_count", 0)) for p in sub)
        teacher_turns = interventions + accepts

        sub_turn_counts = [int(p.get("num_turns_executed", 0)) for p in sub]
        sub_success_turn_counts = [int(p.get("num_turns_executed", 0)) for p in sub if p.get("success")]
        sub_failure_turn_counts = [int(p.get("num_turns_executed", 0)) for p in sub if not p.get("success")]

        sub_verifier_calls = sum(int(p.get("verifier_calls", 0)) for p in sub)
        sub_verifier_blocks = sum(int(p.get("verifier_blocks", 0)) for p in sub)
        sub_verifier_correct = sum(int(p.get("verifier_correct", 0)) for p in sub)
        sub_verifier_incorrect = sum(int(p.get("verifier_incorrect", 0)) for p in sub)
        sub_verifier_unclear = sum(int(p.get("verifier_unclear", 0)) for p in sub)

        by_variant[v] = {
            "count": n,
            "successes": s,
            "accuracy": (s / n) if n else 0.0,
            "over_intervention_rate_all": (oi / n) if n else 0.0,
            "over_intervention_rate_given_student_correct": (oi_gc / len(sub_with_correct)) if sub_with_correct else 0.0,
            "false_acceptance_sessions": fa_sess,
            "false_acceptance_rate": (fa_sess / n) if n else 0.0,
            "false_acceptance_count_total": fa_cnt,
            "interventions_per_session": (interventions / n) if n else 0.0,
            "accepts_per_session": (accepts / n) if n else 0.0,
            "intervention_turn_rate": (interventions / teacher_turns) if teacher_turns else 0.0,
            "avg_turns_per_session": mean_or_zero(sub_turn_counts),
            "avg_turns_success": mean_or_zero(sub_success_turn_counts),
            "avg_turns_failure": mean_or_zero(sub_failure_turn_counts),
            "verifier_calls": sub_verifier_calls,
            "verifier_blocks": sub_verifier_blocks,
            "verifier_block_rate": (sub_verifier_blocks / sub_verifier_calls) if sub_verifier_calls else 0.0,
            "verifier_correct": sub_verifier_correct,
            "verifier_incorrect": sub_verifier_incorrect,
            "verifier_unclear": sub_verifier_unclear,
        }

    return {
        "num_sessions": total,
        "successes": successes,
        "accuracy": (successes / total) if total else 0.0,
        "over_intervention_sessions": over_int_sessions,
        "over_intervention_rate_all": (over_int_sessions / total) if total else 0.0,
        "sessions_with_student_correct": len(with_any_correct),
        "over_intervention_rate_given_student_correct": (over_int_given_correct / len(with_any_correct))
        if with_any_correct
        else 0.0,
        "false_acceptance_sessions": false_accept_sessions,
        "false_acceptance_rate": (false_accept_sessions / total) if total else 0.0,
        "false_acceptance_count_total": false_accept_count_total,
        "teacher_context_trim_events_total": teacher_context_trim_events_total,
        "total_interventions": total_interventions,
        "total_accepts": total_accepts,
        "interventions_per_session": (total_interventions / total) if total else 0.0,
        "accepts_per_session": (total_accepts / total) if total else 0.0,
        "intervention_turn_rate": (total_interventions / total_teacher_turns) if total_teacher_turns else 0.0,
        "avg_turns_per_session": mean_or_zero(turn_counts),
        "avg_turns_success": mean_or_zero(success_turn_counts),
        "avg_turns_failure": mean_or_zero(failure_turn_counts),
        "verifier_calls": verifier_calls_total,
        "verifier_blocks": verifier_blocks_total,
        "verifier_block_rate": (verifier_blocks_total / verifier_calls_total) if verifier_calls_total else 0.0,
        "verifier_correct": verifier_correct_total,
        "verifier_incorrect": verifier_incorrect_total,
        "verifier_unclear": verifier_unclear_total,
        "action_counts": action_counter,
        "final_action_counts": final_action_counter,
        "verifier_override_counts": verifier_override_counter,
        "by_variant": by_variant,
    }


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate DICE policy-conditioned tutoring dialogues.")

    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--result_root", type=str, default=DEFAULT_RES_ROOT)
    parser.add_argument("--errors_jsonl", type=str, default="")

    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--teacher_max_new_tokens", type=int, default=256)
    parser.add_argument("--student_max_new_tokens", type=int, default=256)
    parser.add_argument("--teacher_temperature", type=float, default=0.2)
    parser.add_argument("--student_temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL)
    parser.add_argument("--policy_base_model", type=str, default=POLICY_BASE_MODEL)
    parser.add_argument("--policy_adapter_path", type=str, required=True)

    parser.add_argument("--teacher_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--teacher_pipeline_parallel_size", type=int, default=3)
    parser.add_argument("--student_device", type=int, default=3)
    parser.add_argument("--policy_device", type=int, default=3)
    parser.add_argument("--teacher_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--teacher_max_num_seqs", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="auto")

    parser.add_argument("--policy_use_4bit", action="store_true")
    parser.add_argument("--policy_bf16", action="store_true")
    parser.add_argument("--policy_score_normalization", type=str, default="mean", choices=["mean", "sum"])

    parser.add_argument("--dice_map", action="store_true")
    parser.add_argument("--dice_cost", action="store_true")
    parser.add_argument("--dice_threshold", action="store_true")
    parser.add_argument("--intervention_cost_lambda", type=float, default=0.0)
    parser.add_argument("--intervention_threshold_tau", type=float, default=0.5)

    parser.add_argument(
        "--use_acceptance_verifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use verifier before realizing accept_and_advance. Default: on.",
    )
    parser.add_argument(
        "--acceptance_override_action",
        type=str,
        default="ask_for_justification",
        choices=ACTIVE_ACTIONS,
        help="Action used when verifier blocks accept_and_advance.",
    )
    parser.add_argument(
        "--verifier_max_new_tokens",
        type=int,
        default=8,
        help="Max tokens for verifier output.",
    )

    parser.add_argument("--target", action="store_true")
    parser.add_argument("--easy", action="store_true")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of triplets (not variants) to simulate.",
    )

    args = parser.parse_args()

    dice_mode = resolve_dice_mode(args)
    variants = resolve_variants(args)

    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 0:
        raise RuntimeError("At least one visible GPU is required.")

    teacher_world_size = args.teacher_tensor_parallel_size * args.teacher_pipeline_parallel_size
    if teacher_world_size >= visible_gpus:
        raise ValueError("Teacher TP*PP must be < visible GPUs so one GPU remains for student/policy.")

    if args.student_device < 0 or args.student_device >= visible_gpus:
        raise ValueError(f"--student_device={args.student_device} out of range for visible GPUs={visible_gpus}")
    if args.policy_device < 0 or args.policy_device >= visible_gpus:
        raise ValueError(f"--policy_device={args.policy_device} out of range for visible GPUs={visible_gpus}")

    if args.student_device < teacher_world_size:
        raise ValueError(
            f"student_device={args.student_device} overlaps with teacher GPU slice [0..{teacher_world_size - 1}]."
        )
    if args.policy_device < teacher_world_size:
        raise ValueError(
            f"policy_device={args.policy_device} overlaps with teacher GPU slice [0..{teacher_world_size - 1}]."
        )

    cfg = AutoConfig.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher_heads = int(getattr(cfg, "num_attention_heads", 0))
    if teacher_heads <= 0:
        raise ValueError("Could not infer teacher model num_attention_heads from config.")
    if teacher_heads % args.teacher_tensor_parallel_size != 0:
        raise ValueError(
            f"Teacher TP={args.teacher_tensor_parallel_size} is invalid for attention_heads={teacher_heads}."
        )

    rows = load_json(Path(args.input_json))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        rows = rows[: args.limit]
        print(f"Using first {len(rows)} triplets (--limit={args.limit}).")

    print("Loading student model...")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    student_device = torch.device(f"cuda:{args.student_device}")
    student_torch_dtype = torch.float16 if args.dtype in {"float16", "half"} else torch.bfloat16
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        torch_dtype=student_torch_dtype,
    )
    student_model.to(student_device)
    student_model.eval()

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

    teacher_sampling_params = SamplingParams(
        temperature=args.teacher_temperature,
        top_p=args.top_p,
        max_tokens=args.teacher_max_new_tokens,
    )

    verifier_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.verifier_max_new_tokens,
    )

    teacher_slug = model_slug(args.teacher_model)
    student_slug = model_slug(args.student_model)
    policy_slug = model_slug(args.policy_base_model)
    adapter_slug = Path(args.policy_adapter_path).name

    if dice_mode == "dice_cost":
        mode_tag = f"dice_cost_lam{args.intervention_cost_lambda:g}"
    elif dice_mode == "dice_threshold":
        mode_tag = f"dice_threshold_tau{args.intervention_threshold_tau:g}"
    else:
        mode_tag = "dice_map"

    verifier_tag = "verifier_on" if args.use_acceptance_verifier else "verifier_off"
    mode_tag = f"{mode_tag}_{verifier_tag}"

    combo_slug = f"{mode_tag}__policy_{policy_slug}_{adapter_slug}__gen_{teacher_slug}__student_{student_slug}"

    pred_dir = Path(args.output_root) / combo_slug
    res_dir = Path(args.result_root) / combo_slug
    pred_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    variant_tag = "_".join(variants)
    pred_path = pred_dir / f"preds_dice_sim_{variant_tag}.jsonl"
    res_path = res_dir / f"metrics_dice_sim_{variant_tag}.json"
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else (pred_dir / f"errors_dice_sim_{variant_tag}.jsonl")
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        for variant in variants:
            q = row.get(f"{variant}_question", "")
            a = row.get(f"{variant}_answer", "")
            s = row.get(f"{variant}_steps", "")
            if not q:
                continue
            jobs.append(
                {
                    "index": i,
                    "variant": variant,
                    "question": q,
                    "gold_answer": a,
                    "steps": s,
                    "source_problem_id": row.get("source_problem_id"),
                    "target_skill": row.get("target_skill"),
                }
            )

    predictions: List[Dict[str, Any]] = []
    failed_jobs = 0

    for job in tqdm(jobs, desc=f"Simulating {mode_tag}", unit="sample"):
        try:
            session = simulate_dialogue_dice(
                teacher_llm=teacher_llm,
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                student_device=student_device,
                policy_model=policy_model,
                policy_tokenizer=policy_tokenizer,
                policy_device=policy_device,
                question=job["question"],
                answer=job["gold_answer"],
                steps=job["steps"],
                max_turns=args.max_turns,
                teacher_sampling_params=teacher_sampling_params,
                student_max_new_tokens=args.student_max_new_tokens,
                student_temperature=args.student_temperature,
                top_p=args.top_p,
                dice_mode=dice_mode,
                intervention_cost_lambda=args.intervention_cost_lambda,
                intervention_threshold_tau=args.intervention_threshold_tau,
                policy_score_normalization=args.policy_score_normalization,
                use_acceptance_verifier=args.use_acceptance_verifier,
                acceptance_override_action=args.acceptance_override_action,
                verifier_sampling_params=verifier_sampling_params,
            )

            predictions.append(
                {
                    "index": job["index"],
                    "variant": job["variant"],
                    "source_problem_id": job["source_problem_id"],
                    "target_skill": job["target_skill"],
                    "question": job["question"],
                    "gold_answer": job["gold_answer"],
                    "success": session["success"],
                    "success_turn": session["success_turn"],
                    "first_correct_turn": session["first_correct_turn"],
                    "end_reason": session["end_reason"],
                    "over_intervention": session["over_intervention"],
                    "over_intervention_count": session["over_intervention_count"],
                    "false_acceptance": session["false_acceptance"],
                    "false_acceptance_count": session["false_acceptance_count"],
                    "false_acceptance_turns": session["false_acceptance_turns"],
                    "any_student_correct": session["any_student_correct"],
                    "teacher_context_trim_events": session["teacher_context_trim_events"],
                    "num_turns_executed": session["num_turns_executed"],
                    "intervention_count": session["intervention_count"],
                    "accept_count": session["accept_count"],
                    "verifier_calls": session["verifier_calls"],
                    "verifier_blocks": session["verifier_blocks"],
                    "verifier_correct": session["verifier_correct"],
                    "verifier_incorrect": session["verifier_incorrect"],
                    "verifier_unclear": session["verifier_unclear"],
                    "action_trace": session["action_trace"],
                    "dialogue": session["dialogue"],
                }
            )

        except Exception as exc:
            failed_jobs += 1
            err = {
                "index": job["index"],
                "variant": job["variant"],
                "source_problem_id": job.get("source_problem_id"),
                "target_skill": job.get("target_skill"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(errors_path, err)
            continue

    with pred_path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize_predictions(predictions, variants)

    metrics = {
        "dice_mode": dice_mode,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "policy_base_model": args.policy_base_model,
        "policy_adapter_path": args.policy_adapter_path,
        "variants": variants,
        "input_json": args.input_json,
        "max_turns": args.max_turns,
        "num_jobs_total": len(jobs),
        "num_jobs_failed": failed_jobs,
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
            "policy_device": args.policy_device,
            "teacher_gpu_memory_utilization": args.teacher_gpu_memory_utilization,
            "teacher_max_num_seqs": args.teacher_max_num_seqs,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "policy_use_4bit": args.policy_use_4bit,
            "policy_bf16": args.policy_bf16,
            "policy_score_normalization": args.policy_score_normalization,
            "intervention_cost_lambda": args.intervention_cost_lambda,
            "intervention_threshold_tau": args.intervention_threshold_tau,
            "use_acceptance_verifier": args.use_acceptance_verifier,
            "acceptance_override_action": args.acceptance_override_action,
            "verifier_max_new_tokens": args.verifier_max_new_tokens,
        },
    }

    with res_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {pred_path}")
    print(f"Saved metrics to:     {res_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DEFAULT_INPUT = "data/triplets_250.json"
DEFAULT_PRED_ROOT = "outputs/preds"
DEFAULT_RES_ROOT = "outputs/metrics"

BRIDGE_CODE_DIR = str(
    Path(__file__).resolve().parents[2] / "third_party" / "stratl"
)

ACCEPT_PATTERNS = [
    r"(?i)\b(correct|exactly|that'?s right|well done|great job|excellent|perfect|you got it)\b",
    r"(?i)\b(yes|right)[!.,]",
    r"(?i)now.{0,20}(next|move on|let'?s continue)",
    r"(?i)\b(confirm|indeed|absolutely|precisely)\b",
    r"(?i)\bcorrect\w*[!.]",
    r"(?i)you'?ve.{0,30}(done|calculated|solved|understood|got)",
    r"(?i)keep up the good work",
]


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


def resolve_variants(args: argparse.Namespace) -> List[str]:
    requested: List[str] = []
    if args.target:
        requested.append("target")
    if args.easy:
        requested.append("easy")
    if args.hard:
        requested.append("hard")
    return requested or ["target"]


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


def build_student_system_prompt(question: str) -> str:
    return (
        "You are a student solving a math problem with a teacher.\n"
        "Think internally before answering, but do not reveal chain-of-thought.\n"
        "Answer in about 50 words with brief justification.\n"
        f"\n[Problem]\n{question}\n"
    )


def build_student_prompt_text(student_tokenizer: Any, question: str, dialogue: List[Dict[str, Any]]) -> str:
    messages: List[Dict[str, str]] = [{"role": "system", "content": build_student_system_prompt(question)}]
    for msg in dialogue:
        role = "user" if msg["speaker"] == "teacher" else "assistant"
        messages.append({"role": role, "content": msg["text"]})
    return student_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def generate_one_vllm(llm: LLM, conversation: List[Dict[str, str]], sampling_params: SamplingParams) -> str:
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
    kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": tokenizer.eos_token_id}
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **kwargs)
    prompt_len = encoded["input_ids"].shape[1]
    return tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()


def build_assessor_prompt(pb: str, sol: str, tutor_messages: List[str], student_messages: List[str], assessment_history: List[str]) -> List[Dict[str, str]]:
    purpose = (
        f"A student and their tutor are working on a math problem:\n"
        f"*Problem Statement*:\n##\n{pb}\n##\n\n"
        f"The *Provided Solution* of this problem is:\n##\n{sol}\n##\n\n"
        "The tutor's utterances are preceded by \"Tutor:\" and the student's utterances are preceded by \"Student:\".\n"
        "Analyze the last student's utterance and select all applicable labels from a,b,c,d,e,f,g,h,i,j,k,l,m.\n"
        "Return JSON with keys justification and selection (letters only, no spaces/commas).\n"
    )
    prompt: List[Dict[str, str]] = [{"role": "system", "content": purpose}]
    hist_len = min(len(assessment_history), max(0, len(tutor_messages) - 1), max(0, len(student_messages) - 1))
    for i in range(hist_len):
        prompt.append({"role": "user", "content": f"Tutor: \"{tutor_messages[i]}\"\nStudent: \"{student_messages[i]}\"\n"})
        prompt.append({"role": "assistant", "content": assessment_history[i]})
    prompt.append({"role": "user", "content": f"Tutor: \"{tutor_messages[-1]}\"\nStudent: \"{student_messages[-1]}\"\n"})
    return prompt


def parse_assessment_json(raw: str) -> str:
    s = raw.find("{")
    e = raw.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return json.dumps({"justification": "", "selection": ""}, ensure_ascii=False)
    data = json.loads(raw[s : e + 1])
    selection = str(data.get("selection", ""))
    data["selection"] = selection.replace(" ", "").replace(",", "")
    return json.dumps(data, ensure_ascii=False)


def is_overflow_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "maximum context length" in msg
        or "input tokens" in msg
        or "maximum model length" in msg
        or "prompt (length" in msg
        or "prompt + requested output tokens" in msg
        or "vllmvalidationerror" in msg
    )


def simulate_dialogue_bridge(
    teacher_llm: LLM,
    student_model: Any,
    student_tokenizer: Any,
    student_device: torch.device,
    question: str,
    answer: Any,
    steps: Any,
    max_turns: int,
    teacher_sampling_params: SamplingParams,
    assessor_sampling_params: SamplingParams,
    student_max_new_tokens: int,
    student_temperature: float,
    top_p: float,
    intent_selector: Any,
    prompt_generator: Any,
) -> Dict[str, Any]:
    dialogue: List[Dict[str, Any]] = []
    tutor_messages: List[str] = []
    student_messages: List[str] = []
    assessment_history: List[str] = []
    intent_trace: List[Dict[str, Any]] = []
    pending_correct_turn: int | None = None
    first_correct_turn: int | None = None
    success = False
    end_reason = "timeout"
    over_intervention_count = 0
    any_student_correct = False
    false_acceptance_count = 0
    false_acceptance_turns: List[int] = []
    teacher_context_trim_events = 0

    # Turn 0: cold student attempt.
    cold_prompt = build_student_prompt_text(
        student_tokenizer=student_tokenizer,
        question=question,
        dialogue=[{"speaker": "teacher", "text": f"Can you solve this problem? {question}"}],
    )
    cold_text = generate_one_hf(student_model, student_tokenizer, cold_prompt, student_device, student_max_new_tokens, student_temperature, top_p)
    cold_correct = is_correct(cold_text, answer)
    dialogue.append({"turn": 0, "speaker": "student", "text": cold_text, "is_correct_answer": cold_correct})
    tutor_messages.append(f"Can you solve this problem? {question}")
    student_messages.append(cold_text)
    if cold_correct:
        any_student_correct = True
        pending_correct_turn = 0
        first_correct_turn = 0

    for turn in range(1, max_turns + 1):
        # Bridge assessor.
        trimmed_pairs = 0
        while True:
            try:
                assess_prompt = build_assessor_prompt(
                    pb=question,
                    sol=str(steps),
                    tutor_messages=tutor_messages[trimmed_pairs:],
                    student_messages=student_messages[trimmed_pairs:],
                    assessment_history=assessment_history[trimmed_pairs:],
                )
                raw_assess = generate_one_vllm(teacher_llm, assess_prompt, assessor_sampling_params)
                assessment = parse_assessment_json(raw_assess)
                break
            except Exception as exc:
                if not is_overflow_error(exc):
                    raise
                if (len(tutor_messages) - trimmed_pairs) <= 1 or (len(student_messages) - trimmed_pairs) <= 1:
                    raise
                trimmed_pairs += 1
                teacher_context_trim_events += 1

        intents = intent_selector.get_intent(assessment, open=True)
        intent_names = [getattr(i, "name", str(i)) for i in intents]

        # Bridge rigid prompt generator.
        trimmed_pairs = 0
        while True:
            try:
                teacher_conv = prompt_generator.get_prompt(
                    question,
                    str(steps),
                    student_messages[trimmed_pairs:],
                    tutor_messages[trimmed_pairs:],
                    intents,
                )
                teacher_text = generate_one_vllm(teacher_llm, teacher_conv, teacher_sampling_params)
                break
            except Exception as exc:
                if not is_overflow_error(exc):
                    raise
                if (len(tutor_messages) - trimmed_pairs) <= 1 or (len(student_messages) - trimmed_pairs) <= 1:
                    raise
                trimmed_pairs += 1
                teacher_context_trim_events += 1

        tutor_messages.append(teacher_text)
        dialogue.append({"turn": turn, "speaker": "teacher", "text": teacher_text})
        assessment_history.append(assessment)
        intent_trace.append({"turn": turn, "assessment": assessment, "intents": intent_names})

        if teacher_accepts(teacher_text):
            if student_messages and not is_correct(student_messages[-1], answer):
                false_acceptance_count += 1
                false_acceptance_turns.append(turn)

        if pending_correct_turn is not None:
            if teacher_accepts(teacher_text):
                success = True
                end_reason = "accepted"
                break
            over_intervention_count += 1
            pending_correct_turn = None

        student_prompt = build_student_prompt_text(student_tokenizer, question, dialogue)
        student_text = generate_one_hf(
            student_model, student_tokenizer, student_prompt, student_device, student_max_new_tokens, student_temperature, top_p
        )
        student_correct = is_correct(student_text, answer)
        student_messages.append(student_text)
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
        "false_acceptance_count": false_acceptance_count,
        "false_acceptance_turns": false_acceptance_turns,
        "false_acceptance": false_acceptance_count > 0,
        "any_student_correct": any_student_correct,
        "teacher_context_trim_events": teacher_context_trim_events,
        "num_turns_executed": max((m["turn"] for m in dialogue), default=0),
        "intent_trace": intent_trace,
        "dialogue": dialogue,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge-style pedagogical steering simulation with fixed teacher/student.")
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--result_root", type=str, default=DEFAULT_RES_ROOT)
    parser.add_argument("--errors_jsonl", type=str, default="")
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--teacher_max_new_tokens", type=int, default=256)
    parser.add_argument("--student_max_new_tokens", type=int, default=192)
    parser.add_argument("--teacher_temperature", type=float, default=0.2)
    parser.add_argument("--student_temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL)
    parser.add_argument("--teacher_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--teacher_pipeline_parallel_size", type=int, default=3)
    parser.add_argument("--student_device", type=int, default=3)
    parser.add_argument("--teacher_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--teacher_max_num_seqs", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--target", action="store_true")
    parser.add_argument("--easy", action="store_true")
    parser.add_argument("--hard", action="store_true")
    args = parser.parse_args()

    import sys

    if BRIDGE_CODE_DIR not in sys.path:
        sys.path.append(BRIDGE_CODE_DIR)
    import IntentSelector  # type: ignore
    import PromptGenerator  # type: ignore

    variants = resolve_variants(args)
    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 0:
        raise RuntimeError("At least one visible GPU is required.")
    teacher_world_size = args.teacher_tensor_parallel_size * args.teacher_pipeline_parallel_size
    if teacher_world_size >= visible_gpus:
        raise ValueError("Teacher TP*PP must be < visible GPUs so one GPU remains for student.")
    if args.student_device < teacher_world_size or args.student_device >= visible_gpus:
        raise ValueError("Invalid student_device for current teacher TP/PP setup.")

    cfg = AutoConfig.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher_heads = int(getattr(cfg, "num_attention_heads", 0))
    if teacher_heads <= 0 or teacher_heads % args.teacher_tensor_parallel_size != 0:
        raise ValueError("Invalid teacher tensor parallel size for model attention heads.")

    rows = load_json(Path(args.input_json))
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    student_device = torch.device(f"cuda:{args.student_device}")
    student_dtype = torch.float16 if args.dtype in {"float16", "half"} else torch.bfloat16
    student_model = AutoModelForCausalLM.from_pretrained(args.student_model, trust_remote_code=True, torch_dtype=student_dtype)
    student_model.to(student_device)
    student_model.eval()

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
    assessor_sampling_params = SamplingParams(temperature=0.0, top_p=0.1, max_tokens=300)

    teacher_slug = model_slug(args.teacher_model)
    student_slug = model_slug(args.student_model)
    combo_slug = f"bridge_rule_policy__gen_{teacher_slug}__student_{student_slug}"
    pred_dir = Path(args.output_root) / combo_slug
    res_dir = Path(args.result_root) / combo_slug
    pred_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    variant_tag = "_".join(variants)
    pred_path = pred_dir / f"preds_bridge_sim_{variant_tag}.jsonl"
    res_path = res_dir / f"metrics_bridge_sim_{variant_tag}.json"
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else (pred_dir / f"errors_bridge_sim_{variant_tag}.jsonl")
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        for variant in variants:
            q = row.get(f"{variant}_question", "")
            a = row.get(f"{variant}_answer", "")
            s = row.get(f"{variant}_steps", "")
            if q:
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
    for job in tqdm(jobs, desc="Simulating bridge policy", unit="sample"):
        try:
            intent_selector = IntentSelector.IntentSelector(intent_history=[])
            prompt_generator = PromptGenerator.RigidPromptGenerator()
            session = simulate_dialogue_bridge(
                teacher_llm=teacher_llm,
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                student_device=student_device,
                question=job["question"],
                answer=job["gold_answer"],
                steps=job["steps"],
                max_turns=args.max_turns,
                teacher_sampling_params=teacher_sampling_params,
                assessor_sampling_params=assessor_sampling_params,
                student_max_new_tokens=args.student_max_new_tokens,
                student_temperature=args.student_temperature,
                top_p=args.top_p,
                intent_selector=intent_selector,
                prompt_generator=prompt_generator,
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
                    "intent_trace": session["intent_trace"],
                    "dialogue": session["dialogue"],
                }
            )
        except Exception as exc:
            failed_jobs += 1
            append_jsonl(
                errors_path,
                {
                    "index": job["index"],
                    "variant": job["variant"],
                    "source_problem_id": job.get("source_problem_id"),
                    "target_skill": job.get("target_skill"),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    with pred_path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(predictions)
    successes = sum(1 for p in predictions if p["success"])
    with_any_correct = [p for p in predictions if p["any_student_correct"]]
    over_int_sessions = sum(1 for p in predictions if p["over_intervention"])
    false_accept_sessions = sum(1 for p in predictions if p.get("false_acceptance", False))
    false_accept_count_total = sum(int(p.get("false_acceptance_count", 0)) for p in predictions)
    teacher_context_trim_events_total = sum(int(p.get("teacher_context_trim_events", 0)) for p in predictions)

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
        by_variant[v] = {
            "count": n,
            "successes": s,
            "accuracy": (s / n) if n else 0.0,
            "over_intervention_rate_all": (oi / n) if n else 0.0,
            "over_intervention_rate_given_student_correct": (oi_gc / len(sub_with_correct)) if sub_with_correct else 0.0,
            "false_acceptance_sessions": fa_sess,
            "false_acceptance_rate": (fa_sess / n) if n else 0.0,
            "false_acceptance_count_total": fa_cnt,
        }

    metrics = {
        "policy_type": "bridge_rule_policy",
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "variants": variants,
        "input_json": args.input_json,
        "max_turns": args.max_turns,
        "num_jobs_total": len(jobs),
        "num_jobs_failed": failed_jobs,
        "num_sessions": total,
        "successes": successes,
        "accuracy": (successes / total) if total else 0.0,
        "over_intervention_sessions": over_int_sessions,
        "over_intervention_rate_all": (over_int_sessions / total) if total else 0.0,
        "sessions_with_student_correct": len(with_any_correct),
        "false_acceptance_sessions": false_accept_sessions,
        "false_acceptance_rate": (false_accept_sessions / total) if total else 0.0,
        "false_acceptance_count_total": false_accept_count_total,
        "teacher_context_trim_events_total": teacher_context_trim_events_total,
        "by_variant": by_variant,
        "predictions_path": str(pred_path),
        "errors_path": str(errors_path),
    }
    with res_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {pred_path}")
    print(f"Saved metrics to:     {res_path}")


if __name__ == "__main__":
    main()

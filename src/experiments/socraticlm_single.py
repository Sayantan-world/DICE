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


TEACHER_MODEL = "CogBase-USTC/SocraticLM"
STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DEFAULT_INPUT = "data/triplets_250.json"
DEFAULT_PRED_ROOT = "outputs/preds"
DEFAULT_RES_ROOT = "outputs/metrics"

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


def build_teacher_system_prompt(question: str, answer: Any, steps: str) -> str:
    _ = (question, answer, steps)
    return (
        "You are a Socratic teacher, please guide me to solve the [Problem] "
        "with heuristic questions based on the following information.\n"
    )


def build_student_system_prompt(question: str) -> str:
    return (
        "You are a student solving a math problem with a teacher.\n"
        "Think internally before answering, but never reveal internal reasoning.\n"
        "Answer in about 50 words and explain briefly why.\n"
        "If you are confident, provide a final numeric answer.\n"
        f"\n[Problem]\n{question}\n"
    )


def build_teacher_conversation(
    question: str,
    answer: Any,
    steps: str,
    dialogue: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    # Keep labels close to the model card format. We map dataset steps to [Analysis].
    user_seed = (
        f"[Problem] {question} "
        f"[Answer] {answer} "
        f"[Analysis] {steps}"
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_teacher_system_prompt(question, answer, steps)},
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
    return messages


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


def simulate_dialogue(
    teacher_llm: LLM,
    student_model: Any,
    student_tokenizer: Any,
    student_device: torch.device,
    question: str,
    answer: Any,
    steps: str,
    max_turns: int,
    teacher_sampling_params: SamplingParams,
    student_max_new_tokens: int,
    student_temperature: float,
    top_p: float,
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
        # Guard against context-overflow crashes by trimming oldest history and retrying.
        trimmed_dialogue = dialogue
        while True:
            teacher_conversation = build_teacher_conversation(
                question=question,
                answer=answer,
                steps=steps,
                dialogue=trimmed_dialogue,
            )
            try:
                teacher_text = generate_one_vllm(teacher_llm, teacher_conversation, teacher_sampling_params)
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
                # If not a context-length issue, surface the real error.
                if not overflow:
                    raise
                # If no history left to trim, re-raise.
                if len(trimmed_dialogue) <= 1:
                    raise
                # Drop earliest two messages (typically one teacher-student pair).
                trimmed_dialogue = trimmed_dialogue[2:]
                teacher_context_trim_events += 1
        dialogue.append({"turn": turn, "speaker": "teacher", "text": teacher_text})

        if teacher_accepts(teacher_text):
            last_student = [m for m in dialogue if m.get("speaker") == "student"]
            if last_student and not last_student[-1].get("is_correct_answer", False):
                false_acceptance_count += 1
                false_acceptance_turns.append(turn)

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

    # Edge case: if the last student response was correct but no next teacher turn remains.
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
        "dialogue": dialogue,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Socratic teacher-student dialogues with fixed models.")
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--result_root", type=str, default=DEFAULT_RES_ROOT)
    parser.add_argument("--errors_jsonl", type=str, default="", help="Optional path to append per-sample errors.")
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--teacher_max_new_tokens", type=int, default=160)
    parser.add_argument("--student_max_new_tokens", type=int, default=96)
    parser.add_argument("--teacher_temperature", type=float, default=0.2)
    parser.add_argument("--student_temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL)
    parser.add_argument("--teacher_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--teacher_pipeline_parallel_size", type=int, default=3)
    parser.add_argument("--student_device", type=int, default=3, help="GPU index for student HF model (e.g., 3).")
    parser.add_argument("--teacher_gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--teacher_max_num_seqs", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="auto")

    parser.add_argument("--target", action="store_true", help="Use target question variant.")
    parser.add_argument("--easy", action="store_true", help="Use easy question variant.")
    parser.add_argument("--hard", action="store_true", help="Use hard question variant.")
    args = parser.parse_args()

    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 0:
        raise RuntimeError("At least one visible GPU is required.")
    if args.teacher_tensor_parallel_size < 1 or args.teacher_pipeline_parallel_size < 1:
        raise ValueError("--teacher_tensor_parallel_size and --teacher_pipeline_parallel_size must be >= 1")
    if args.student_device < 0 or args.student_device >= visible_gpus:
        raise ValueError(
            f"--student_device={args.student_device} is out of range for visible GPUs={visible_gpus}"
        )
    teacher_world_size = args.teacher_tensor_parallel_size * args.teacher_pipeline_parallel_size
    if teacher_world_size >= visible_gpus:
        raise ValueError("Teacher TP*PP must be < visible GPUs so one GPU remains for the student model.")
    if args.student_device < teacher_world_size:
        raise ValueError(
            f"student_device={args.student_device} overlaps with teacher GPU slice [0..{teacher_world_size - 1}]. "
            "Choose a student_device outside teacher TP*PP range."
        )

    cfg = AutoConfig.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher_heads = int(getattr(cfg, "num_attention_heads", 0))
    if teacher_heads <= 0:
        raise ValueError("Could not infer teacher model num_attention_heads from config.")
    if teacher_heads % args.teacher_tensor_parallel_size != 0:
        raise ValueError(
            f"Teacher TP={args.teacher_tensor_parallel_size} is invalid for attention_heads={teacher_heads}. "
            "Use a TP that divides the number of heads (e.g., 1, 2, 4, 7, 14 for this model)."
        )
    variants = resolve_variants(args)
    rows = load_json(Path(args.input_json))

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    teacher_llm = build_llm(
        model_name=args.teacher_model,
        tensor_parallel_size=args.teacher_tensor_parallel_size,
        pipeline_parallel_size=args.teacher_pipeline_parallel_size,
        gpu_memory_utilization=args.teacher_gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.teacher_max_num_seqs,
        dtype=args.dtype,
    )
    student_device = torch.device(f"cuda:{args.student_device}")
    student_torch_dtype = (
        torch.float16 if args.dtype in {"float16", "half"} else torch.bfloat16
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        torch_dtype=student_torch_dtype,
    )
    student_model.to(student_device)
    student_model.eval()

    teacher_sampling_params = SamplingParams(
        temperature=args.teacher_temperature,
        top_p=args.top_p,
        max_tokens=args.teacher_max_new_tokens,
    )
    teacher_slug = model_slug(args.teacher_model)
    student_slug = model_slug(args.student_model)
    combo_slug = f"{teacher_slug}__vs__{student_slug}"

    pred_dir = Path(args.output_root) / combo_slug
    res_dir = Path(args.result_root) / combo_slug
    pred_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    variant_tag = "_".join(variants)
    pred_path = pred_dir / f"preds_socratic_sim_{variant_tag}.jsonl"
    res_path = res_dir / f"metrics_socratic_sim_{variant_tag}.json"
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else (pred_dir / f"errors_socratic_sim_{variant_tag}.jsonl")
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
    for job in tqdm(jobs, desc="Simulating dialogues", unit="sample"):
        try:
            session = simulate_dialogue(
                teacher_llm=teacher_llm,
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                student_device=student_device,
                question=job["question"],
                answer=job["gold_answer"],
                steps=job["steps"],
                max_turns=args.max_turns,
                teacher_sampling_params=teacher_sampling_params,
                student_max_new_tokens=args.student_max_new_tokens,
                student_temperature=args.student_temperature,
                top_p=args.top_p,
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
            with errors_path.open("a", encoding="utf-8") as ef:
                ef.write(json.dumps(err, ensure_ascii=False) + "\n")
            continue

    with pred_path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(predictions)
    successes = sum(1 for p in predictions if p["success"])
    with_any_correct = [p for p in predictions if p["any_student_correct"]]
    over_int_sessions = sum(1 for p in predictions if p["over_intervention"])
    over_int_given_correct = sum(1 for p in with_any_correct if p["over_intervention"])
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
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "variants": variants,
        "input_json": args.input_json,
        "max_turns": args.max_turns,
        "num_sessions": total,
        "num_jobs_total": len(jobs),
        "num_jobs_failed": failed_jobs,
        "successes": successes,
        "accuracy": (successes / total) if total else 0.0,
        "over_intervention_sessions": over_int_sessions,
        "over_intervention_rate_all": (over_int_sessions / total) if total else 0.0,
        "false_acceptance_sessions": false_accept_sessions,
        "false_acceptance_rate": (false_accept_sessions / total) if total else 0.0,
        "false_acceptance_count_total": false_accept_count_total,
        "teacher_context_trim_events_total": teacher_context_trim_events_total,
        "sessions_with_student_correct": len(with_any_correct),
        "over_intervention_rate_given_student_correct": (over_int_given_correct / len(with_any_correct))
        if with_any_correct
        else 0.0,
        "by_variant": by_variant,
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
            "teacher_gpu_memory_utilization": args.teacher_gpu_memory_utilization,
            "teacher_max_num_seqs": args.teacher_max_num_seqs,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
        },
    }

    with res_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {pred_path}")
    print(f"Saved metrics to:     {res_path}")


if __name__ == "__main__":
    main()

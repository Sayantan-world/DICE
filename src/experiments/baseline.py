from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_INPUT = "data/triplets_250.json"
DEFAULT_PRED_ROOT = "outputs/preds"
DEFAULT_RES_ROOT = "outputs/metrics"


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of datapoints.")
    return data


def resolve_mode(args: argparse.Namespace) -> str:
    selected = [args.zeroshot, args.cot, args.steps_augmented_cot]
    if sum(1 for x in selected if x) > 1:
        raise ValueError("Choose only one mode from --zeroshot, --cot, --steps_augmented_cot")
    if args.cot:
        return "cot"
    if args.steps_augmented_cot:
        return "steps_augmented_cot"
    return "zeroshot"


def resolve_variants(args: argparse.Namespace) -> List[str]:
    requested = []
    if args.target:
        requested.append("target")
    if args.easy:
        requested.append("easy")
    if args.hard:
        requested.append("hard")
    # Default behavior: only target when nothing is explicitly requested.
    return requested or ["target"]


def build_prompt_text(mode: str, question: str, steps: str) -> str:
    if mode == "zeroshot":
        return (
            "Solve the following question.\n"
            "Output STRICTLY a valid JSON object with exactly one key: "
            "{\"answer\": \"...\"}\n\n"
            f"Question:\n{question}"
        )
    if mode == "cot":
        return (
            "Think step by step internally, but do NOT output reasoning.\n"
            "Output STRICTLY a valid JSON object with exactly one key: "
            "{\"answer\": \"...\"}\n\n"
            f"Question:\n{question}"
        )
    if mode == "steps_augmented_cot":
        return (
            "Think step by step internally and use the provided steps if helpful, "
            "but do NOT output reasoning.\n"
            "Output STRICTLY a valid JSON object with exactly one key: "
            "{\"answer\": \"...\"}\n\n"
            f"Question:\n{question}\n\n"
            f"Here are the steps:\n{steps}\n"
        )
    raise ValueError(f"Unsupported mode: {mode}")


def extract_answer_fallback(text: str) -> str:
    raw = text.strip()
    if not raw:
        return ""
    patterns = [
        r"(?i)final answer\s*[:\-]\s*(.+)",
        r"(?i)answer\s*[:\-]\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw)
        if matches:
            return matches[-1].strip().splitlines()[0].strip()
    return raw.splitlines()[-1].strip()


def extract_answer_from_structured_text(text: str) -> str:
    raw = text.strip()
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"]).strip()
    except json.JSONDecodeError:
        pass
    # fallback: try to recover inline JSON-like answer field
    m = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
    if m:
        return m.group(1).strip()
    return extract_answer_fallback(raw)


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


def is_malformed_structured_output(raw_text: str, pred_answer: str) -> bool:
    raw = raw_text.strip()
    if not raw:
        return True
    if raw in {"{", "{}", '{"answer":', "{\"answer\":"}:
        return True
    if not pred_answer or pred_answer.strip() in {"{", "}", '"answer":', "answer"}:
        return True
    if "answer" not in raw.lower():
        return True
    return False


def is_correct(pred: str, gold: Any) -> bool:
    pred_norm, pred_num = normalize_answer(pred)
    gold_norm, gold_num = normalize_answer(str(gold))
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6
    return pred_norm == gold_norm


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot baseline inference on target/easy/hard variants.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--result_root", type=str, default=DEFAULT_RES_ROOT)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=0,
        help="vLLM tensor parallel size; 0 means auto-use all visible GPUs.",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="vLLM pipeline parallel size.",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="auto")

    parser.add_argument("--target", action="store_true", help="Solve target questions.")
    parser.add_argument("--easy", action="store_true", help="Solve easy questions.")
    parser.add_argument("--hard", action="store_true", help="Solve hard questions.")

    parser.add_argument("--zeroshot", action="store_true", help="Question only.")
    parser.add_argument("--cot", action="store_true", help='Question + system prompt "Think step by step."')
    parser.add_argument(
        "--steps_augmented_cot",
        action="store_true",
        help="Question + steps + CoT system prompt.",
    )
    args = parser.parse_args()

    mode = resolve_mode(args)
    variants = resolve_variants(args)

    rows = load_json(Path(args.input_json))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 0:
        raise RuntimeError("vLLM requires at least one visible GPU.")
    requested_tp = args.tensor_parallel_size if args.tensor_parallel_size > 0 else visible_gpus
    requested_pp = max(1, args.pipeline_parallel_size)

    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    num_heads = int(getattr(cfg, "num_attention_heads", 0))
    if num_heads <= 0:
        raise ValueError("Could not infer num_attention_heads from model config.")

    tp = min(requested_tp, visible_gpus)
    pp = requested_pp
    if num_heads % tp != 0:
        divisors = [d for d in range(min(num_heads, visible_gpus), 0, -1) if num_heads % d == 0]
        adjusted_tp = divisors[0]
        requested_total = min(visible_gpus, max(1, requested_tp * requested_pp))
        adjusted_pp = max(1, min(visible_gpus // adjusted_tp, requested_total // adjusted_tp))
        if adjusted_tp * adjusted_pp > visible_gpus:
            adjusted_pp = max(1, visible_gpus // adjusted_tp)
        print(
            f"[vLLM] Adjusting parallelism for model heads={num_heads}: "
            f"requested TP={requested_tp}, PP={requested_pp} -> TP={adjusted_tp}, PP={adjusted_pp}"
        )
        tp, pp = adjusted_tp, adjusted_pp

    if tp * pp > visible_gpus:
        raise ValueError(f"TP*PP={tp * pp} exceeds visible GPUs={visible_gpus}.")

    llm = LLM(
        model=args.model_name,
        tokenizer=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_chunked_prefill=True,
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    structured = StructuredOutputsParams(
        json=schema,
        disable_any_whitespace=True,
        disable_additional_properties=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        structured_outputs=structured,
    )

    slug = model_slug(args.model_name)
    pred_dir = Path(args.output_root) / slug
    res_dir = Path(args.result_root) / slug
    pred_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    variant_tag = "_".join(variants)
    pred_path = pred_dir / f"preds_{mode}_{variant_tag}.jsonl"
    res_path = res_dir / f"metrics_{mode}_{variant_tag}.json"

    jobs: List[Dict[str, Any]] = []
    prompts: List[str] = []
    for i, row in enumerate(rows):
        for variant in variants:
            q = row.get(f"{variant}_question", "")
            s = row.get(f"{variant}_steps", "")
            gold = row.get(f"{variant}_answer", "")
            if not q:
                continue
            prompt_text = build_prompt_text(mode=mode, question=q, steps=s)
            jobs.append(
                {
                    "job_index": len(prompts),
                    "index": i,
                    "variant": variant,
                    "source_problem_id": row.get("source_problem_id"),
                    "target_skill": row.get("target_skill"),
                    "gold_answer": gold,
                }
            )
            prompts.append(prompt_text)

    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    predictions: List[Dict[str, Any]] = []
    for job, output in tqdm(zip(jobs, outputs), total=len(jobs), desc="Scoring outputs", unit="pred"):
        raw_output = output.outputs[0].text.strip()
        pred_answer = extract_answer_from_structured_text(raw_output)
        if is_malformed_structured_output(raw_output, pred_answer):
            retry_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max(args.max_new_tokens, 384),
                structured_outputs=structured,
            )
            retry_out = llm.generate(
                [prompts[job["job_index"]]],
                sampling_params=retry_params,
                use_tqdm=False,
            )[0]
            retry_raw = retry_out.outputs[0].text.strip()
            retry_pred = extract_answer_from_structured_text(retry_raw)
            if not is_malformed_structured_output(retry_raw, retry_pred):
                raw_output = retry_raw
                pred_answer = retry_pred
        predictions.append(
            {
                "index": job["index"],
                "variant": job["variant"],
                "source_problem_id": job["source_problem_id"],
                "target_skill": job["target_skill"],
                "gold_answer": job["gold_answer"],
                "pred_answer": pred_answer,
                "raw_output": raw_output,
                "correct": is_correct(pred_answer, job["gold_answer"]),
            }
        )

    with pred_path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Accuracy report
    by_variant: Dict[str, Dict[str, Any]] = {}
    overall_correct = 0
    for v in variants:
        subset = [r for r in predictions if r["variant"] == v]
        c = sum(1 for r in subset if r["correct"])
        n = len(subset)
        by_variant[v] = {"count": n, "correct": c, "accuracy": (c / n) if n else 0.0}
        overall_correct += c

    overall_n = len(predictions)
    metrics = {
        "model_name": args.model_name,
        "mode": mode,
        "variants": variants,
        "input_json": args.input_json,
        "num_predictions": overall_n,
        "overall_accuracy": (overall_correct / overall_n) if overall_n else 0.0,
        "by_variant": by_variant,
        "predictions_path": str(pred_path),
        "backend": "vllm_structured_outputs",
        "vllm_config": {
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "visible_gpus": visible_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    with res_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {pred_path}")
    print(f"Saved metrics to:     {res_path}")


if __name__ == "__main__":
    main()

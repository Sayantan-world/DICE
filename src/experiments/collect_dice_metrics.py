#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect DICE simulation metrics from many experiment folders.

Expected structure:
ROOT_DIR/
  exp_1/
    .../metrics_dice_sim_target_easy_hard.json
  exp_2/
    .../metrics_dice_sim_target_easy_hard.json
  ...

Example:
python collect_dice_metrics.py \
  --root_dir outputs/preds \
  --output_json outputs/preds/combined_metrics.json \
  --output_csv outputs/preds/combined_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


METRICS_FILENAME = "metrics_dice_sim_target_easy_hard.json"


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def infer_run_name(metrics_path: Path, root_dir: Path) -> str:
    """
    Builds a compact run name from the folder containing the metrics file.
    """
    try:
        rel = metrics_path.parent.relative_to(root_dir)
        return str(rel).replace("/", "__")
    except Exception:
        return metrics_path.parent.name


def infer_mode_from_path_or_json(data: Dict[str, Any], metrics_path: Path) -> str:
    if data.get("dice_mode"):
        return str(data["dice_mode"])

    path_str = str(metrics_path).lower()
    if "dice_cost" in path_str:
        return "dice_cost"
    if "dice_threshold" in path_str:
        return "dice_threshold"
    if "dice_map" in path_str:
        return "dice_map"
    return "unknown"


def infer_teacher_short_name(teacher_model: str) -> str:
    if not teacher_model:
        return "unknown"

    lower = teacher_model.lower()

    if "socraticlm" in lower or "cogbase-ustc" in lower:
        return "socraticlm"
    if "qwen2.5-7b-instruct" in lower or "qwen/qwen2.5-7b-instruct" in lower:
        return "qwen25_7b_instruct"

    return teacher_model.replace("/", "_").replace(".", "p")


def infer_hyperparam(data: Dict[str, Any], metrics_path: Path) -> Dict[str, Any]:
    config = data.get("config", {}) or {}
    dice_mode = data.get("dice_mode", "")

    lam = config.get("intervention_cost_lambda", data.get("intervention_cost_lambda", None))
    tau = config.get("intervention_threshold_tau", data.get("intervention_threshold_tau", None))

    # Fallback from path if needed
    path_str = str(metrics_path)

    if lam is None:
        m = re.search(r"lambda[_-]?([0-9p.]+)", path_str)
        if m:
            lam = m.group(1).replace("p", ".")

    if tau is None:
        m = re.search(r"tau[_-]?([0-9p.]+)", path_str)
        if m:
            tau = m.group(1).replace("p", ".")

    hp = {
        "intervention_cost_lambda": lam,
        "intervention_threshold_tau": tau,
    }

    if dice_mode == "dice_cost":
        hp["selected_hyperparam"] = lam
        hp["selected_hyperparam_name"] = "lambda"
    elif dice_mode == "dice_threshold":
        hp["selected_hyperparam"] = tau
        hp["selected_hyperparam_name"] = "tau"
    else:
        hp["selected_hyperparam"] = None
        hp["selected_hyperparam_name"] = None

    return hp


def compute_extra_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds useful derived metrics for ranking.
    """
    success = row.get("accuracy")
    avg_turns_success = row.get("avg_turns_success")
    interventions_per_session = row.get("interventions_per_session")
    over_int = row.get("over_intervention_rate_all")
    false_accept = row.get("false_acceptance_rate")

    extra = {}

    # Success per average successful turn: rough efficiency score.
    if success is not None and avg_turns_success not in (None, 0):
        extra["success_per_turn_success"] = success / avg_turns_success
    else:
        extra["success_per_turn_success"] = None

    # A simple calibration penalty: bad if either over-intervention or false acceptance is high.
    if over_int is not None and false_accept is not None:
        extra["calibration_error_sum"] = over_int + false_accept
    else:
        extra["calibration_error_sum"] = None

    # Composite score for quick ranking.
    if (
        success is not None
        and false_accept is not None
        and over_int is not None
        and interventions_per_session is not None
    ):
        extra["dice_composite_score"] = (
            success
            - 0.50 * false_accept
            - 0.50 * over_int
            - 0.05 * interventions_per_session
        )
    else:
        extra["dice_composite_score"] = None

    return extra


def flatten_metrics(data: Dict[str, Any], metrics_path: Path, root_dir: Path) -> Dict[str, Any]:
    config = data.get("config", {}) or {}
    hp = infer_hyperparam(data, metrics_path)

    teacher_model = safe_get(data, "teacher_model")
    teacher_short = infer_teacher_short_name(teacher_model)

    row: Dict[str, Any] = {
        # Identity
        "run_name": infer_run_name(metrics_path, root_dir),
        "metrics_path": str(metrics_path),
        "dice_mode": infer_mode_from_path_or_json(data, metrics_path),
        "teacher_model": teacher_model,
        "teacher_short": teacher_short,
        "student_model": safe_get(data, "student_model"),
        "policy_base_model": safe_get(data, "policy_base_model"),
        "policy_adapter_path": safe_get(data, "policy_adapter_path"),
        "input_json": safe_get(data, "input_json"),
        "variants": ",".join(safe_get(data, "variants", [])) if isinstance(safe_get(data, "variants", []), list) else safe_get(data, "variants"),

        # Hyperparams
        "intervention_cost_lambda": hp.get("intervention_cost_lambda"),
        "intervention_threshold_tau": hp.get("intervention_threshold_tau"),
        "selected_hyperparam_name": hp.get("selected_hyperparam_name"),
        "selected_hyperparam": hp.get("selected_hyperparam"),

        # Core counts
        "max_turns": safe_get(data, "max_turns"),
        "num_jobs_total": safe_get(data, "num_jobs_total"),
        "num_jobs_failed": safe_get(data, "num_jobs_failed"),
        "num_sessions": safe_get(data, "num_sessions"),
        "successes": safe_get(data, "successes"),

        # Headline task metrics
        "accuracy": safe_get(data, "accuracy"),
        "avg_turns_per_session": safe_get(data, "avg_turns_per_session"),
        "avg_turns_success": safe_get(data, "avg_turns_success"),
        "avg_turns_failure": safe_get(data, "avg_turns_failure"),

        # Intervention metrics
        "total_interventions": safe_get(data, "total_interventions"),
        "total_accepts": safe_get(data, "total_accepts"),
        "interventions_per_session": safe_get(data, "interventions_per_session"),
        "accepts_per_session": safe_get(data, "accepts_per_session"),
        "intervention_turn_rate": safe_get(data, "intervention_turn_rate"),

        # Calibration metrics
        "over_intervention_sessions": safe_get(data, "over_intervention_sessions"),
        "over_intervention_rate_all": safe_get(data, "over_intervention_rate_all"),
        "sessions_with_student_correct": safe_get(data, "sessions_with_student_correct"),
        "over_intervention_rate_given_student_correct": safe_get(data, "over_intervention_rate_given_student_correct"),
        "false_acceptance_sessions": safe_get(data, "false_acceptance_sessions"),
        "false_acceptance_rate": safe_get(data, "false_acceptance_rate"),
        "false_acceptance_count_total": safe_get(data, "false_acceptance_count_total"),

        # Verifier metrics
        "use_acceptance_verifier": config.get("use_acceptance_verifier", None),
        "acceptance_override_action": config.get("acceptance_override_action", None),
        "verifier_calls": safe_get(data, "verifier_calls"),
        "verifier_blocks": safe_get(data, "verifier_blocks"),
        "verifier_block_rate": safe_get(data, "verifier_block_rate"),
        "verifier_correct": safe_get(data, "verifier_correct"),
        "verifier_incorrect": safe_get(data, "verifier_incorrect"),
        "verifier_unclear": safe_get(data, "verifier_unclear"),

        # Action counts as compact JSON strings for CSV compatibility
        "action_counts": safe_get(data, "action_counts", {}),
        "final_action_counts": safe_get(data, "final_action_counts", {}),
        "verifier_override_counts": safe_get(data, "verifier_override_counts", {}),

        # Generation/config
        "teacher_temperature": config.get("teacher_temperature", None),
        "student_temperature": config.get("student_temperature", None),
        "top_p": config.get("top_p", None),
        "policy_score_normalization": config.get("policy_score_normalization", None),
    }

    # Variant-specific metrics
    by_variant = safe_get(data, "by_variant", {}) or {}
    for variant_name in ["target", "easy", "hard"]:
        v = by_variant.get(variant_name, {}) or {}
        prefix = f"{variant_name}_"

        for key in [
            "count",
            "successes",
            "accuracy",
            "over_intervention_rate_all",
            "over_intervention_rate_given_student_correct",
            "false_acceptance_sessions",
            "false_acceptance_rate",
            "false_acceptance_count_total",
            "interventions_per_session",
            "accepts_per_session",
            "intervention_turn_rate",
            "avg_turns_per_session",
            "avg_turns_success",
            "avg_turns_failure",
            "verifier_calls",
            "verifier_blocks",
            "verifier_block_rate",
            "verifier_correct",
            "verifier_incorrect",
            "verifier_unclear",
        ]:
            row[prefix + key] = v.get(key, None)

    row.update(compute_extra_metrics(row))

    return row


def find_metrics_files(root_dir: Path) -> List[Path]:
    return sorted(root_dir.rglob(METRICS_FILENAME))


def make_json_safe(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures JSON serializability.
    """
    out = {}
    for k, v in row.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def write_json(rows: List[Dict[str, Any]], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump([make_json_safe(r) for r in rows], f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote combined JSON: {output_json}")


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    if not rows:
        print("[WARN] No rows to write CSV.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Stable key ordering: keys from first row, then any extra keys
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for row in rows:
            csv_row = {}
            for k in keys:
                v = row.get(k, None)
                if isinstance(v, (dict, list)):
                    csv_row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    csv_row[k] = v
            writer.writerow(csv_row)

    print(f"[OK] Wrote combined CSV: {output_csv}")


def print_best_runs(rows: List[Dict[str, Any]], top_k: int = 10) -> None:
    if not rows:
        return

    def sort_key(r: Dict[str, Any]):
        score = r.get("dice_composite_score")
        if score is None:
            return -999
        return score

    ranked = sorted(rows, key=sort_key, reverse=True)

    print("\nTop runs by dice_composite_score:")
    print("-" * 120)
    print(
        f"{'rank':<5} {'mode':<15} {'teacher':<20} {'hp':<8} "
        f"{'acc':>7} {'turns_s':>8} {'int/sess':>9} {'over':>8} {'false_acc':>10} {'score':>8}"
    )
    print("-" * 120)

    for i, r in enumerate(ranked[:top_k], start=1):
        hp_name = r.get("selected_hyperparam_name")
        hp_val = r.get("selected_hyperparam")
        hp = "-" if hp_name is None else f"{hp_name}={hp_val}"

        print(
            f"{i:<5} "
            f"{str(r.get('dice_mode')):<15} "
            f"{str(r.get('teacher_short')):<20} "
            f"{str(hp):<8} "
            f"{fmt_float(r.get('accuracy')):>7} "
            f"{fmt_float(r.get('avg_turns_success')):>8} "
            f"{fmt_float(r.get('interventions_per_session')):>9} "
            f"{fmt_float(r.get('over_intervention_rate_all')):>8} "
            f"{fmt_float(r.get('false_acceptance_rate')):>10} "
            f"{fmt_float(r.get('dice_composite_score')):>8}"
        )

    print("-" * 120)


def fmt_float(x: Any, digits: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect DICE simulation metrics from experiment folders."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing many experiment subdirectories.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save combined metrics JSON.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save combined metrics CSV.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top runs to print by composite score.",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else None

    metrics_files = find_metrics_files(root_dir)

    print(f"[INFO] Root dir: {root_dir}")
    print(f"[INFO] Found {len(metrics_files)} metrics files named {METRICS_FILENAME}")

    rows: List[Dict[str, Any]] = []

    for path in metrics_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            row = flatten_metrics(data, path, root_dir)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to parse {path}: {e}")

    # Sort for readability
    rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("teacher_short")),
            str(r.get("dice_mode")),
            str(r.get("selected_hyperparam_name")),
            str(r.get("selected_hyperparam")),
            str(r.get("run_name")),
        ),
    )

    write_json(rows, output_json)

    if output_csv:
        write_csv(rows, output_csv)

    print_best_runs(rows, top_k=args.top_k)

    print(f"\n[INFO] Parsed runs: {len(rows)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Offline IV sensitivity analysis.

Recomputes Intervention Value from stored judge labels. No student rollouts
and no judge API calls. Used for the descriptive HELPFUL / NEUTRAL / HARMFUL
audit; policy training uses the numerical IV directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SCORE_MAP: Dict[str, float] = {
    "helpful_progress": 1.0,
    "partial_progress": 0.5,
    "no_progress_or_harm": 0.0,
}

SCORE_MAPS: Dict[str, Dict[str, float]] = {
    "reported": {
        "helpful_progress": 1.0,
        "partial_progress": 0.5,
        "no_progress_or_harm": 0.0,
    },
    "binary": {
        "helpful_progress": 1.0,
        "partial_progress": 0.0,
        "no_progress_or_harm": 0.0,
    },
    "partial_0.25": {
        "helpful_progress": 1.0,
        "partial_progress": 0.25,
        "no_progress_or_harm": 0.0,
    },
    "harm_-0.5": {
        "helpful_progress": 1.0,
        "partial_progress": 0.5,
        "no_progress_or_harm": -0.5,
    },
    "harm_-1": {
        "helpful_progress": 1.0,
        "partial_progress": 0.5,
        "no_progress_or_harm": -1.0,
    },
    "signed_ternary": {
        "helpful_progress": 1.0,
        "partial_progress": 0.0,
        "no_progress_or_harm": -1.0,
    },
}

DEFAULT_EPSILONS = (0.0, 0.05, 0.10, 0.125, 0.25, 0.375, 0.50)
FLOAT_EPS = 1e-6
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def parse_epsilons(raw: str) -> List[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def parse_named_inputs(items: Sequence[str]) -> List[Tuple[str, Path]]:
    named: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Expected NAME=PATH for --input, got {item!r}"
            )
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --input {item!r}")
        named.append((name, Path(path)))
    return named


def round6(x: float) -> float:
    return round(float(x), 6)


def pct(x: int, n: int) -> float:
    return round((x / n) * 100.0, 3) if n > 0 else 0.0


def mean_safe(xs: Sequence[float]) -> float:
    return round6(sum(xs) / len(xs)) if xs else 0.0


def std_safe(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return round6(statistics.pstdev(xs))


def quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return round6(ys[0])
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return round6(ys[lo])
    w = pos - lo
    return round6(ys[lo] * (1.0 - w) + ys[hi] * w)


def labels_from_eval(eval_blob: Dict[str, Any]) -> List[str]:
    rollouts = eval_blob.get("rollouts") or []
    labels = [str(r.get("judge_label")) for r in rollouts]
    if labels:
        return labels
    counts = eval_blob.get("judge_label_counts") or {}
    expanded: List[str] = []
    for label, count in counts.items():
        expanded.extend([str(label)] * int(count))
    return expanded


def mu_from_labels(labels: Sequence[str], score_map: Dict[str, float]) -> float:
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        if label not in score_map:
            raise KeyError(f"Unknown judge label {label!r}")
        scores.append(float(score_map[label]))
    return sum(scores) / len(scores)


def partition(ivs: Sequence[float], epsilon: float) -> Dict[str, Any]:
    n = len(ivs)
    thr = max(float(epsilon), FLOAT_EPS) if epsilon <= 0 else float(epsilon)
    helpful = sum(1 for x in ivs if x > thr)
    harmful = sum(1 for x in ivs if x < -thr)
    neutral = n - helpful - harmful
    return {
        "epsilon": epsilon,
        "threshold_used": thr,
        "n": n,
        "helpful_n": helpful,
        "neutral_n": neutral,
        "harmful_n": harmful,
        "helpful_pct": pct(helpful, n),
        "neutral_pct": pct(neutral, n),
        "harmful_pct": pct(harmful, n),
    }


def value_counts(ivs: Sequence[float]) -> List[Dict[str, Any]]:
    counter: Counter[float] = Counter(round6(x) for x in ivs)
    n = len(ivs)
    rows = []
    for value in sorted(counter):
        count = counter[value]
        rows.append({
            "iv": value,
            "count": count,
            "share": round(count / n, 6) if n else 0.0,
            "share_pct": pct(count, n),
        })
    return rows


def histogram(ivs: Sequence[float], bin_width: float, lo: float, hi: float) -> Dict[str, Any]:
    n = len(ivs)
    edges = []
    x = lo
    while x < hi - 1e-12:
        edges.append(round6(x))
        x += bin_width
    edges.append(round6(hi))
    counts = [0] * (len(edges) - 1)
    underflow = 0
    overflow = 0
    for v in ivs:
        if v < lo - 1e-12:
            underflow += 1
            continue
        if v > hi + 1e-12:
            overflow += 1
            continue
        idx = int(math.floor((v - lo) / bin_width + 1e-12))
        if idx >= len(counts):
            idx = len(counts) - 1
        if idx < 0:
            underflow += 1
            continue
        counts[idx] += 1
    rows = []
    for i, count in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        rows.append({
            "bin_left": left,
            "bin_right": right,
            "bin_center": round6(0.5 * (left + right)),
            "count": count,
            "share": round(count / n, 6) if n else 0.0,
            "share_pct": pct(count, n),
        })
    return {
        "bin_width": bin_width,
        "lo": lo,
        "hi": hi,
        "underflow": underflow,
        "overflow": overflow,
        "bins": rows,
    }


def summarize_ivs(ivs: Sequence[float]) -> Dict[str, Any]:
    n = len(ivs)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean": mean_safe(ivs),
        "std": std_safe(ivs),
        "min": round6(min(ivs)),
        "max": round6(max(ivs)),
        "quantiles": {f"p{int(q * 100):02d}": quantile(ivs, q) for q in QUANTILES},
        "share_gt_0": pct(sum(1 for x in ivs if x > FLOAT_EPS), n),
        "share_eq_0": pct(sum(1 for x in ivs if abs(x) <= FLOAT_EPS), n),
        "share_lt_0": pct(sum(1 for x in ivs if x < -FLOAT_EPS), n),
        "share_abs_gt_0.125": pct(sum(1 for x in ivs if abs(x) > 0.125 + FLOAT_EPS), n),
        "share_abs_gt_0.25": pct(sum(1 for x in ivs if abs(x) > 0.25 + FLOAT_EPS), n),
        "share_abs_gt_0.5": pct(sum(1 for x in ivs if abs(x) > 0.5 + FLOAT_EPS), n),
    }


def load_compact_rows(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            action_labels = labels_from_eval(rec.get("action_eval") or {})
            accept_labels = labels_from_eval(rec.get("accept_eval") or {})
            rows.append({
                "gold_action": rec.get("gold_action", "MISSING"),
                "variant_name": rec.get("variant_name"),
                "provenance": rec.get("provenance"),
                "stored_mu_action": float(rec.get("mu_action", 0.0)),
                "stored_mu_accept": float(rec.get("mu_accept_and_advance", 0.0)),
                "stored_iv": float(rec.get("intervention_value", 0.0)),
                "action_labels": action_labels,
                "accept_labels": accept_labels,
            })
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def reconstruct(rows: Sequence[Dict[str, Any]], score_map: Dict[str, float]) -> List[float]:
    ivs = []
    for row in rows:
        mu_a = mu_from_labels(row["action_labels"], score_map)
        mu_b = mu_from_labels(row["accept_labels"], score_map)
        ivs.append(mu_a - mu_b)
    return ivs


def reconstruction_check(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mismatches = 0
    max_abs = 0.0
    missing_labels = 0
    for row in rows:
        if not row["action_labels"] or not row["accept_labels"]:
            missing_labels += 1
            continue
        mu_a = mu_from_labels(row["action_labels"], DEFAULT_SCORE_MAP)
        mu_b = mu_from_labels(row["accept_labels"], DEFAULT_SCORE_MAP)
        iv = mu_a - mu_b
        err = abs(iv - float(row["stored_iv"]))
        max_abs = max(max_abs, err)
        if err > 1e-8:
            mismatches += 1
    return {
        "n": len(rows),
        "missing_label_rows": missing_labels,
        "mismatches_vs_stored_iv": mismatches,
        "max_abs_error": round(max_abs, 12),
        "ok": missing_labels == 0 and mismatches == 0,
    }


def subset_rows(rows: Sequence[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    if name == "all":
        return list(rows)
    if name == "non_accept":
        return [r for r in rows if r["gold_action"] != "accept_and_advance"]
    if name == "accept_only":
        return [r for r in rows if r["gold_action"] == "accept_and_advance"]
    raise KeyError(name)


def label_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    overall: Counter[str] = Counter()
    action_arm: Counter[str] = Counter()
    accept_arm: Counter[str] = Counter()
    for row in rows:
        action_arm.update(row["action_labels"])
        accept_arm.update(row["accept_labels"])
    overall.update(action_arm)
    overall.update(accept_arm)
    return {
        "overall": dict(overall),
        "action_arm": dict(action_arm),
        "accept_arm": dict(accept_arm),
    }


def by_action_partitions(
    rows: Sequence[Dict[str, Any]],
    ivs: Sequence[float],
    epsilons: Sequence[float],
) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row, iv in zip(rows, ivs):
        grouped[str(row["gold_action"])].append(iv)
    out: Dict[str, Any] = {}
    for action in sorted(grouped):
        action_ivs = grouped[action]
        out[action] = {
            "n": len(action_ivs),
            "mean_iv": mean_safe(action_ivs),
            "median_iv": round6(statistics.median(action_ivs)) if action_ivs else 0.0,
            "by_epsilon": {str(eps): partition(action_ivs, eps) for eps in epsilons},
        }
    return out


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_plot(
    out_dir: Path,
    dataset: str,
    ivs: Sequence[float],
    epsilon_rows: Sequence[Dict[str, Any]],
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped ({e})")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    counts = Counter(round6(x) for x in ivs)
    xs = sorted(counts)
    ys = [counts[x] for x in xs]
    axes[0].bar(xs, ys, width=0.08, color="#3b6ea5", edgecolor="black", linewidth=0.3)
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title(f"{dataset}: IV distribution")
    axes[0].set_xlabel("IV")
    axes[0].set_ylabel("count")

    eps = [r["epsilon"] for r in epsilon_rows]
    helpful = [r["helpful_pct"] for r in epsilon_rows]
    neutral = [r["neutral_pct"] for r in epsilon_rows]
    harmful = [r["harmful_pct"] for r in epsilon_rows]
    axes[1].plot(eps, helpful, marker="o", label="HELPFUL")
    axes[1].plot(eps, neutral, marker="s", label="NEUTRAL")
    axes[1].plot(eps, harmful, marker="^", label="HARMFUL")
    axes[1].set_title("Partition vs epsilon (reported map)")
    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("percent")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    path = out_dir / f"{dataset}_iv_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def fmt_map(score_map: Dict[str, float]) -> str:
    return (
        f"helpful={score_map['helpful_progress']:g}, "
        f"partial={score_map['partial_progress']:g}, "
        f"harm={score_map['no_progress_or_harm']:g}"
    )


def markdown_report(
    dataset: str,
    n_all: int,
    recon: Dict[str, Any],
    reported_summary: Dict[str, Any],
    reported_partition0: Dict[str, Any],
    epsilon_table: List[Dict[str, Any]],
    mapping_table: List[Dict[str, Any]],
    mapping_eps_table: List[Dict[str, Any]],
) -> str:
    p0 = reported_partition0
    nearly_half = (
        f"{p0['helpful_n']}/{p0['n']} = {p0['helpful_pct']:.1f}%"
    )
    lines = [
        f"# IV sensitivity: `{dataset}`",
        "",
        "HELPFUL / NEUTRAL / HARMFUL and the margin ε are **descriptive only**.",
        "Policy training uses the numerical IV directly.",
        "",
        "## Reconstruction check",
        "",
        f"- rows: {n_all}",
        f"- missing label rows: {recon['missing_label_rows']}",
        f"- mismatches vs stored IV under the reported map: {recon['mismatches_vs_stored_iv']}",
        f"- max abs error: {recon['max_abs_error']}",
        f"- ok: {recon['ok']}",
        "",
        "## Reported configuration",
        "",
        "- score map: helpful=1.0, partial=0.5, harm=0.0",
        "- partition at ε=0 (threshold 1e-6)",
        f"- n = {reported_summary['n']}",
        f"- mean IV = {reported_summary['mean']:.4f}",
        f"- median IV = {reported_summary['quantiles']['p50']:.4f}",
        f"- HELPFUL (IV > 0): **{nearly_half}**",
        f"- NEUTRAL: {p0['neutral_n']} ({p0['neutral_pct']:.1f}%)",
        f"- HARMFUL: {p0['harmful_n']} ({p0['harmful_pct']:.1f}%)",
        "",
        "The “nearly half” claim is **specific to this configuration**.",
        "",
        "## IV distribution (reported map)",
        "",
        "| stat | value |",
        "|---|---|",
        f"| n | {reported_summary['n']} |",
        f"| mean | {reported_summary['mean']:.4f} |",
        f"| std | {reported_summary['std']:.4f} |",
        f"| min | {reported_summary['min']:.4f} |",
        f"| p05 | {reported_summary['quantiles']['p05']:.4f} |",
        f"| p25 | {reported_summary['quantiles']['p25']:.4f} |",
        f"| median | {reported_summary['quantiles']['p50']:.4f} |",
        f"| p75 | {reported_summary['quantiles']['p75']:.4f} |",
        f"| p95 | {reported_summary['quantiles']['p95']:.4f} |",
        f"| max | {reported_summary['max']:.4f} |",
        f"| P(IV>0) | {reported_summary['share_gt_0']:.1f}% |",
        f"| P(IV=0) | {reported_summary['share_eq_0']:.1f}% |",
        f"| P(IV<0) | {reported_summary['share_lt_0']:.1f}% |",
        f"| P(|IV|>0.125) | {reported_summary['share_abs_gt_0.125']:.1f}% |",
        f"| P(|IV|>0.25) | {reported_summary['share_abs_gt_0.25']:.1f}% |",
        "",
        "## ε sweep (reported map, all rows)",
        "",
        "| ε | HELPFUL n | HELPFUL % | NEUTRAL % | HARMFUL % |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in epsilon_table:
        if row["subset"] != "all" or row["mapping"] != "reported":
            continue
        lines.append(
            f"| {row['epsilon']:g} | {row['helpful_n']} | {row['helpful_pct']:.1f} | "
            f"{row['neutral_pct']:.1f} | {row['harmful_pct']:.1f} |"
        )

    lines.extend([
        "",
        "## Score-map sweep at ε=0 (all rows)",
        "",
        "| map | scores | mean IV | HELPFUL % | NEUTRAL % | HARMFUL % |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in mapping_table:
        if row["subset"] != "all" or abs(float(row["epsilon"])) > 1e-12:
            continue
        lines.append(
            f"| {row['mapping']} | {row['score_map']} | {row['mean_iv']:.4f} | "
            f"{row['helpful_pct']:.1f} | {row['neutral_pct']:.1f} | {row['harmful_pct']:.1f} |"
        )

    lines.extend([
        "",
        "## Map × ε (HELPFUL %, all rows)",
        "",
    ])
    maps = []
    eps_vals = []
    for row in mapping_eps_table:
        if row["subset"] != "all":
            continue
        if row["mapping"] not in maps:
            maps.append(row["mapping"])
        if row["epsilon"] not in eps_vals:
            eps_vals.append(row["epsilon"])
    header = "| map | " + " | ".join(f"ε={e:g}" for e in eps_vals) + " |"
    sep = "|" + "---|" * (len(eps_vals) + 1)
    lines.append(header)
    lines.append(sep)
    lookup = {
        (row["mapping"], row["epsilon"]): row["helpful_pct"]
        for row in mapping_eps_table
        if row["subset"] == "all"
    }
    for mapping in maps:
        cells = " | ".join(f"{lookup[(mapping, e)]:.1f}" for e in eps_vals)
        lines.append(f"| {mapping} | {cells} |")

    lines.extend([
        "",
        "## Qualification for the paper",
        "",
        f"Under the reported map and ε=0, {nearly_half} of rows have IV > 0. "
        "This share is configuration-specific: it shrinks as ε increases, and it "
        "moves when partial/harm scores are remapped. Training does not use the "
        "three-level partition; it uses the scalar IV.",
        "",
    ])
    return "\n".join(lines) + "\n"


def analyze_dataset(
    name: str,
    path: Path,
    out_dir: Path,
    epsilons: Sequence[float],
    limit: Optional[int],
) -> Dict[str, Any]:
    print(f"\n=== {name} ===")
    print(f"loading {path}")
    rows = load_compact_rows(path, limit=limit)
    recon = reconstruction_check(rows)
    print(
        f"rows={len(rows)} recon_ok={recon['ok']} "
        f"mismatches={recon['mismatches_vs_stored_iv']} "
        f"max_abs_err={recon['max_abs_error']}"
    )

    subsets = {
        "all": subset_rows(rows, "all"),
        "non_accept": subset_rows(rows, "non_accept"),
        "accept_only": subset_rows(rows, "accept_only"),
    }

    epsilon_csv: List[Dict[str, Any]] = []
    mapping_csv: List[Dict[str, Any]] = []
    mapping_eps_csv: List[Dict[str, Any]] = []
    by_action_csv: List[Dict[str, Any]] = []
    hist_csv: List[Dict[str, Any]] = []

    report: Dict[str, Any] = {
        "dataset": name,
        "input_jsonl": str(path),
        "n_loaded": len(rows),
        "reconstruction": recon,
        "score_maps": {k: dict(v) for k, v in SCORE_MAPS.items()},
        "epsilons": list(epsilons),
        "label_counts": label_counts(rows),
        "subsets": {},
    }

    reported_ivs_all = reconstruct(subsets["all"], SCORE_MAPS["reported"])
    reported_summary = summarize_ivs(reported_ivs_all)
    reported_partition0 = partition(reported_ivs_all, 0.0)
    hist = histogram(reported_ivs_all, bin_width=0.125, lo=-1.0, hi=1.0)
    exact = value_counts(reported_ivs_all)

    for row in exact:
        hist_csv.append({"dataset": name, "subset": "all", "kind": "exact", **row})
    for row in hist["bins"]:
        hist_csv.append({"dataset": name, "subset": "all", "kind": "bin", **row})

    for subset_name, subset in subsets.items():
        subset_block: Dict[str, Any] = {
            "n": len(subset),
            "mappings": {},
        }
        for map_name, score_map in SCORE_MAPS.items():
            ivs = reconstruct(subset, score_map)
            summary = summarize_ivs(ivs)
            parts = {str(eps): partition(ivs, eps) for eps in epsilons}
            subset_block["mappings"][map_name] = {
                "score_map": dict(score_map),
                "summary": summary,
                "exact_value_counts": value_counts(ivs) if map_name == "reported" else None,
                "histogram_0.125": histogram(ivs, 0.125, -2.0, 2.0),
                "by_epsilon": parts,
                "by_action": by_action_partitions(subset, ivs, epsilons) if map_name == "reported" else None,
            }

            for eps in epsilons:
                part = parts[str(eps)]
                rec = {
                    "dataset": name,
                    "subset": subset_name,
                    "mapping": map_name,
                    "score_map": fmt_map(score_map),
                    "epsilon": eps,
                    "n": part["n"],
                    "mean_iv": summary["mean"],
                    "median_iv": summary["quantiles"]["p50"],
                    "helpful_n": part["helpful_n"],
                    "neutral_n": part["neutral_n"],
                    "harmful_n": part["harmful_n"],
                    "helpful_pct": part["helpful_pct"],
                    "neutral_pct": part["neutral_pct"],
                    "harmful_pct": part["harmful_pct"],
                }
                mapping_eps_csv.append(rec)
                if map_name == "reported":
                    epsilon_csv.append(rec)
                if abs(eps) <= 1e-12 or abs(eps - 0.125) <= 1e-12:
                    mapping_csv.append(rec)

            if map_name == "reported":
                by_action = subset_block["mappings"][map_name]["by_action"]
                for action, block in by_action.items():
                    for eps in epsilons:
                        part = block["by_epsilon"][str(eps)]
                        by_action_csv.append({
                            "dataset": name,
                            "subset": subset_name,
                            "action": action,
                            "epsilon": eps,
                            "n": block["n"],
                            "mean_iv": block["mean_iv"],
                            "median_iv": block["median_iv"],
                            "helpful_n": part["helpful_n"],
                            "neutral_n": part["neutral_n"],
                            "harmful_n": part["harmful_n"],
                            "helpful_pct": part["helpful_pct"],
                            "neutral_pct": part["neutral_pct"],
                            "harmful_pct": part["harmful_pct"],
                        })

        report["subsets"][subset_name] = subset_block

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}_sensitivity.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    write_csv(
        out_dir / f"{name}_iv_distribution.csv",
        ["dataset", "subset", "kind", "iv", "count", "share", "share_pct",
         "bin_left", "bin_right", "bin_center"],
        hist_csv,
    )
    write_csv(
        out_dir / f"{name}_epsilon_sweep.csv",
        ["dataset", "subset", "mapping", "score_map", "epsilon", "n", "mean_iv",
         "median_iv", "helpful_n", "neutral_n", "harmful_n", "helpful_pct",
         "neutral_pct", "harmful_pct"],
        epsilon_csv,
    )
    write_csv(
        out_dir / f"{name}_mapping_sweep.csv",
        ["dataset", "subset", "mapping", "score_map", "epsilon", "n", "mean_iv",
         "median_iv", "helpful_n", "neutral_n", "harmful_n", "helpful_pct",
         "neutral_pct", "harmful_pct"],
        mapping_csv,
    )
    write_csv(
        out_dir / f"{name}_mapping_x_epsilon.csv",
        ["dataset", "subset", "mapping", "score_map", "epsilon", "n", "mean_iv",
         "median_iv", "helpful_n", "neutral_n", "harmful_n", "helpful_pct",
         "neutral_pct", "harmful_pct"],
        mapping_eps_csv,
    )
    write_csv(
        out_dir / f"{name}_by_action.csv",
        ["dataset", "subset", "action", "epsilon", "n", "mean_iv", "median_iv",
         "helpful_n", "neutral_n", "harmful_n", "helpful_pct", "neutral_pct",
         "harmful_pct"],
        by_action_csv,
    )

    md = markdown_report(
        dataset=name,
        n_all=len(rows),
        recon=recon,
        reported_summary=reported_summary,
        reported_partition0=reported_partition0,
        epsilon_table=epsilon_csv,
        mapping_table=mapping_csv,
        mapping_eps_table=mapping_eps_csv,
    )
    md_path = out_dir / f"{name}_REPORT.md"
    md_path.write_text(md, encoding="utf-8")

    plot_path = maybe_plot(
        out_dir,
        name,
        reported_ivs_all,
        [r for r in epsilon_csv if r["subset"] == "all" and r["mapping"] == "reported"],
    )

    print(md)
    print(f"saved JSON   {json_path}")
    print(f"saved report {md_path}")
    if plot_path:
        print(f"saved plot   {plot_path}")
    print(
        f"HELPFUL at reported/ε=0: "
        f"{reported_partition0['helpful_n']}/{reported_partition0['n']} "
        f"({reported_partition0['helpful_pct']:.1f}%)"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline IV distribution + ε/score-map sensitivity from stored judge labels.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Repeatable NAME=PATH jsonl. Default is the train-target observed IV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/iv_sensitivity"),
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default=",".join(str(x) for x in DEFAULT_EPSILONS),
        help="Comma-separated ε values for HELPFUL/NEUTRAL/HARMFUL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for a smoke test.",
    )
    args = parser.parse_args()

    if not args.input:
        raise ValueError("Provide at least one --input NAME=PATH jsonl of stored IV records.")
    inputs = parse_named_inputs(args.input)
    epsilons = parse_epsilons(args.epsilons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    for name, path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        ds_dir = args.output_dir / name
        analyze_dataset(name, path, ds_dir, epsilons, args.limit)
        combined.append(name)

    print("\nDone.")
    print("datasets:", ", ".join(combined))
    print("output_dir:", args.output_dir)


if __name__ == "__main__":
    main()

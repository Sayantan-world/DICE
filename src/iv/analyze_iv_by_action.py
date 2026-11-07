from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def pct(x: int, n: int) -> float:
    return round((x / n) * 100.0, 3) if n > 0 else 0.0


def mean_safe(xs: List[float]) -> float:
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def median_safe(xs: List[float]) -> float:
    return round(statistics.median(xs), 6) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate intervention value results by action.")
    parser.add_argument("--input_jsonl", type=Path, required=True, help="IV JSONL file")
    parser.add_argument("--output_json", type=Path, required=True, help="Output JSON summary")
    parser.add_argument("--output_csv", type=Path, required=True, help="Output CSV summary")
    parser.add_argument(
        "--zero_threshold",
        type=float,
        default=1e-6,
        help="Threshold for zeroish IV bucket. Default: 1e-6",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    zero_thr = args.zero_threshold

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        action = row.get("gold_action", "MISSING")
        grouped[action].append(row)

    summary: Dict[str, Any] = {
        "num_rows": len(rows),
        "zero_threshold": zero_thr,
        "by_action": {},
    }

    csv_rows: List[Dict[str, Any]] = []

    for action in sorted(grouped.keys()):
        items = grouped[action]
        n = len(items)

        ivs = [float(r["intervention_value"]) for r in items]
        mu_action = [float(r["mu_action"]) for r in items]
        mu_accept = [float(r["mu_accept_and_advance"]) for r in items]

        pos = sum(1 for x in ivs if x > zero_thr)
        neg = sum(1 for x in ivs if x < -zero_thr)
        zeroish = n - pos - neg

        record = {
            "count": n,
            "mean_iv": mean_safe(ivs),
            "median_iv": median_safe(ivs),
            "min_iv": round(min(ivs), 6) if ivs else 0.0,
            "max_iv": round(max(ivs), 6) if ivs else 0.0,
            "mean_mu_action": mean_safe(mu_action),
            "mean_mu_accept_and_advance": mean_safe(mu_accept),
            "positive_count": pos,
            "zeroish_count": zeroish,
            "negative_count": neg,
            "positive_pct": pct(pos, n),
            "zeroish_pct": pct(zeroish, n),
            "negative_pct": pct(neg, n),
        }

        summary["by_action"][action] = record

        csv_rows.append({
            "action": action,
            **record,
        })

    # overall summary
    all_ivs = [float(r["intervention_value"]) for r in rows]
    all_mu_action = [float(r["mu_action"]) for r in rows]
    all_mu_accept = [float(r["mu_accept_and_advance"]) for r in rows]

    all_pos = sum(1 for x in all_ivs if x > zero_thr)
    all_neg = sum(1 for x in all_ivs if x < -zero_thr)
    all_zeroish = len(rows) - all_pos - all_neg

    summary["overall"] = {
        "count": len(rows),
        "mean_iv": mean_safe(all_ivs),
        "median_iv": median_safe(all_ivs),
        "mean_mu_action": mean_safe(all_mu_action),
        "mean_mu_accept_and_advance": mean_safe(all_mu_accept),
        "positive_count": all_pos,
        "zeroish_count": all_zeroish,
        "negative_count": all_neg,
        "positive_pct": pct(all_pos, len(rows)),
        "zeroish_pct": pct(all_zeroish, len(rows)),
        "negative_pct": pct(all_neg, len(rows)),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "action",
        "count",
        "mean_iv",
        "median_iv",
        "min_iv",
        "max_iv",
        "mean_mu_action",
        "mean_mu_accept_and_advance",
        "positive_count",
        "zeroish_count",
        "negative_count",
        "positive_pct",
        "zeroish_pct",
        "negative_pct",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print("\n=== OVERALL ===")
    for k, v in summary["overall"].items():
        print(f"{k}: {v}")

    print("\n=== BY ACTION ===")
    for action, rec in summary["by_action"].items():
        print(
            f"{action:24s} "
            f"n={rec['count']:4d} "
            f"mean_iv={rec['mean_iv']:.4f} "
            f"pos={rec['positive_pct']:.1f}% "
            f"zero={rec['zeroish_pct']:.1f}% "
            f"neg={rec['negative_pct']:.1f}%"
        )

    print(f"\nSaved JSON to: {args.output_json}")
    print(f"Saved CSV to:  {args.output_csv}")


if __name__ == "__main__":
    main()
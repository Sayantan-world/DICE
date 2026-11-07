#!/usr/bin/env python3
"""Download DICE-Bench triplets, policy annotations, and IV estimates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HF_DATASET = "chocokiddo/DICE-Bench"
TRIPLET_FILES = {
    "triplets_250": "triplets_250.json",
    "triplets_all": "triplets_all.json",
}
POLICY_FILES = {
    "train": "train.jsonl",
    "validation": "val.jsonl",
    "test": "test.jsonl",
}
IV_FILES = {
    "target_observed": "target_observed.jsonl",
    "easy_synthetic": "easy_synthetic.jsonl",
    "hard_synthetic": "hard_synthetic.jsonl",
}


def _to_plain(value):
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_plain(v) for v in value]
    return value


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_triplets(dataset: str, split: str, dest: Path, force: bool) -> Path:
    if dest.exists() and not force:
        print(f"skip {dest} (already exists)", file=sys.stderr)
        return dest
    from datasets import load_dataset

    rows = [_to_plain(dict(ex)) for ex in load_dataset(dataset, split=split)]
    _write_json(dest, rows)
    print(f"wrote {len(rows)} rows -> {dest}", file=sys.stderr)
    return dest


def download_policy(dataset: str, output_dir: Path, force: bool) -> Path:
    out = output_dir / "policy"
    if (out / "train.jsonl").exists() and not force:
        print(f"skip {out} (already exists)", file=sys.stderr)
        return out

    from datasets import load_dataset

    ds = load_dataset(dataset, "policy")
    all_rows = []
    for split, filename in POLICY_FILES.items():
        rows = [_to_plain(dict(ex)) for ex in ds[split]]
        _write_jsonl(out / filename, rows)
        all_rows.extend(rows)
        print(f"wrote {len(rows)} rows -> {out / filename}", file=sys.stderr)

    _write_jsonl(out / "all.jsonl", all_rows)
    _write_json(
        out / "split_meta.json",
        {
            "seed": 42,
            "train_rows": len(ds["train"]),
            "val_rows": len(ds["validation"]),
            "test_rows": len(ds["test"]),
            "train_problems": len({r["source_problem_id"] for r in ds["train"]}),
            "val_problems": len({r["source_problem_id"] for r in ds["validation"]}),
            "test_problems": len({r["source_problem_id"] for r in ds["test"]}),
        },
    )
    print(f"wrote {len(all_rows)} rows -> {out / 'all.jsonl'}", file=sys.stderr)
    return out


def download_iv(dataset: str, output_dir: Path, force: bool) -> Path:
    out = output_dir / "iv"
    if (out / "target_observed.jsonl").exists() and not force:
        print(f"skip {out} (already exists)", file=sys.stderr)
        return out

    from datasets import load_dataset

    ds = load_dataset(dataset, "iv")
    for split, filename in IV_FILES.items():
        rows = []
        for ex in ds[split]:
            row = _to_plain(dict(ex))
            if row.get("action_eval_json") and not row.get("action_eval"):
                row["action_eval"] = json.loads(row["action_eval_json"])
            if row.get("accept_eval_json") and not row.get("accept_eval"):
                row["accept_eval"] = json.loads(row["accept_eval_json"])
            rows.append(row)
        _write_jsonl(out / filename, rows)
        print(f"wrote {len(rows)} rows -> {out / filename}", file=sys.stderr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DICE-Bench data from Hugging Face.")
    parser.add_argument("--dataset", default=HF_DATASET)
    parser.add_argument(
        "--split",
        choices=["triplets_250", "triplets_all", "triplets", "policy", "iv", "all"],
        default="triplets",
        help="triplets_250 / triplets_all / triplets (both eval JSON) / policy / iv / all",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-path", action="store_true")
    args = parser.parse_args()

    last = None
    want_triplets = args.split in {"triplets_250", "triplets_all", "triplets", "all"}
    if want_triplets:
        names = list(TRIPLET_FILES) if args.split in {"triplets", "all"} else [args.split]
        for name in names:
            last = download_triplets(
                args.dataset, name, args.output_dir / TRIPLET_FILES[name], args.force
            )
    if args.split in {"policy", "all"}:
        last = download_policy(args.dataset, args.output_dir, args.force)
    if args.split in {"iv", "all"}:
        last = download_iv(args.dataset, args.output_dir, args.force)

    if args.print_path and last is not None:
        print(last)


if __name__ == "__main__":
    main()

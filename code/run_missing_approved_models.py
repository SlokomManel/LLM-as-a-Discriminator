#!/usr/bin/env python3
"""
Backfill missing/underpowered experiment cells for approved models only.

Approved models:
- groq / llama-3.1-8b-instant
- google / gemini-2.5-flash
- google / gemini-2.0-flash

A cell is a dataset x synthetic_method x condition tuple.
For each approved model, this script counts valid verdicts (REAL/SYNTHETIC)
across existing JSONL files and optionally runs additional batches until a
minimum target valid count is reached.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# Local import from code/
sys.path.insert(0, str(Path(__file__).parent))
import evaluate  # noqa: E402

VALID_LABELS = {"REAL", "SYNTHETIC"}
METHODS = ["ctgan", "tvae", "gaussian_copula"]
CONDS = ["C1", "C2"]

APPROVED_MODELS: List[Tuple[str, str]] = [
    ("groq", "llama-3.1-8b-instant"),
    ("google", "gemini-2.5-flash"),
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    real_csv: Path
    synthetic_dir: Path
    results_dir: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_datasets(root: Path) -> Dict[str, DatasetConfig]:
    return {
        "adult": DatasetConfig(
            name="adult",
            real_csv=root / "data" / "real" / "adult.csv",
            synthetic_dir=root / "data" / "synthetic",
            results_dir=root / "results" / "adult",
        ),
        "census": DatasetConfig(
            name="census",
            real_csv=root / "data" / "census" / "real" / "census.csv",
            synthetic_dir=root / "data" / "census" / "synthetic",
            results_dir=root / "results" / "census",
        ),
    }


def sanitize_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model)


def iter_result_jsonl(results_dir: Path) -> Iterable[Path]:
    if not results_dir.exists():
        return []
    return sorted(p for p in results_dir.glob("*.jsonl") if "human" not in p.name)


def count_valid_cells(results_dir: Path, provider: str, model: str) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {(m, c): 0 for m in METHODS for c in CONDS}
    for path in iter_result_jsonl(results_dir):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("provider") != provider or rec.get("model") != model:
                    continue
                method = rec.get("synthetic_method")
                cond = rec.get("condition")
                if method not in METHODS or cond not in CONDS:
                    continue
                if rec.get("predicted_label") in VALID_LABELS:
                    counts[(method, cond)] += 1
    return counts


def load_synthetic_df(synth_dir: Path, method: str) -> pd.DataFrame:
    path = synth_dir / method / "synthetic.csv"
    if not path.exists():
        raise FileNotFoundError(f"Synthetic CSV not found for method '{method}': {path}")
    return pd.read_csv(path)


def run_batch(
    ds_cfg: DatasetConfig,
    provider: str,
    model: str,
    method: str,
    condition: str,
    n_trials: int,
    n_rows_shown: int,
    max_quota_retries: int,
) -> bool:
    """Run a batch. Returns True if the batch completed without quota errors."""
    real_df = pd.read_csv(ds_cfg.real_csv)
    synthetic_df = load_synthetic_df(ds_cfg.synthetic_dir, method)

    evaluate.RESULTS_DIR = ds_cfg.results_dir
    evaluate.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = evaluate.run_experiment_single_method(
        real_df=real_df,
        synthetic_df=synthetic_df,
        method_name=method,
        provider=provider,
        model=model,
        n_trials=n_trials,
        conditions=[condition],
        n_rows_shown=n_rows_shown,
        sample_rows=None,
        output_file=None,
        resume=False,
        max_quota_retries=max_quota_retries,
    )

    # If every trial errored (likely quota), signal failure.
    valid = {"REAL", "SYNTHETIC"}
    all_errored = results and all(r.get("predicted_label") not in valid for r in results)
    if all_errored:
        print(f"[QUOTA] {method:15s} {condition} -> all trials returned ERROR; quota likely exhausted. Skipping cell.")
        return False
    return True


def plan_and_backfill(
    ds_cfg: DatasetConfig,
    provider: str,
    model: str,
    target_valid: int,
    batch_size: int,
    max_batches: int,
    n_rows_shown: int,
    max_quota_retries: int,
    dry_run: bool,
) -> None:
    print(f"\n=== {ds_cfg.name.upper()} | {provider} | {model} ===")

    for method in METHODS:
        for cond in CONDS:
            batches_used = 0
            while True:
                counts = count_valid_cells(ds_cfg.results_dir, provider, model)
                current = counts[(method, cond)]
                gap = max(0, target_valid - current)

                if gap == 0:
                    print(f"[OK] {method:15s} {cond} -> valid={current} (target={target_valid})")
                    break

                if batches_used >= max_batches:
                    print(
                        f"[STOP] {method:15s} {cond} -> valid={current}, "
                        f"target={target_valid}, max_batches={max_batches} reached"
                    )
                    break

                # Keep n_trials even because queue uses n_trials//2 per label.
                n_trials = max(batch_size, gap)
                if n_trials % 2 == 1:
                    n_trials += 1

                print(
                    f"[RUN] {method:15s} {cond} -> valid={current}, gap={gap}, "
                    f"batch_trials={n_trials}, max_quota_retries={max_quota_retries}"
                )

                if dry_run:
                    batches_used += 1
                    # Simulate single step in dry-run to avoid infinite loops.
                    break

                ok = run_batch(
                    ds_cfg=ds_cfg,
                    provider=provider,
                    model=model,
                    method=method,
                    condition=cond,
                    n_trials=n_trials,
                    n_rows_shown=n_rows_shown,
                    max_quota_retries=max_quota_retries,
                )
                batches_used += 1
                if not ok:
                    # Quota hit — stop this cell immediately.
                    break
                time.sleep(1)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Backfill missing approved-model experiment cells")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "census"],
        choices=["adult", "census"],
        help="Datasets to process",
    )
    parser.add_argument(
        "--target-valid",
        type=int,
        default=10,
        help="Minimum valid verdicts required per method x condition cell",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Minimum trials to run per backfill batch (per cell)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=3,
        help="Max backfill batches per cell",
    )
    parser.add_argument(
        "--n-rows-shown",
        type=int,
        default=20,
        help="Rows shown to model in each prompt",
    )
    parser.add_argument(
        "--max-quota-retries",
        type=int,
        default=0,
        help="Max retry attempts on quota errors per trial (default 0 = fail-fast, no sleeping)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print missing plan only; do not run API calls",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=root,
        help="Project root path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ds_map = default_datasets(args.root)
    for ds_name in args.datasets:
        ds_cfg = ds_map[ds_name]
        if not ds_cfg.real_csv.exists():
            print(f"[ERROR] Real CSV missing: {ds_cfg.real_csv}")
            return 1
        if not ds_cfg.synthetic_dir.exists():
            print(f"[ERROR] Synthetic dir missing: {ds_cfg.synthetic_dir}")
            return 1
        ds_cfg.results_dir.mkdir(parents=True, exist_ok=True)

    print("Approved models:")
    for provider, model in APPROVED_MODELS:
        print(f"- {provider} / {model}")

    print(
        f"\nSettings: target_valid={args.target_valid}, batch_size={args.batch_size}, "
        f"max_batches={args.max_batches}, max_quota_retries={args.max_quota_retries}, dry_run={args.dry_run}"
    )

    for ds_name in args.datasets:
        ds_cfg = ds_map[ds_name]
        for provider, model in APPROVED_MODELS:
            plan_and_backfill(
                ds_cfg=ds_cfg,
                provider=provider,
                model=model,
                target_valid=args.target_valid,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                n_rows_shown=args.n_rows_shown,
                max_quota_retries=args.max_quota_retries,
                dry_run=args.dry_run,
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

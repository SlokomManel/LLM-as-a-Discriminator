#!/usr/bin/env python3
"""
Archive invalid/unapproved LLM result files.

Default behavior is DRY-RUN.
Use --execute to actually move files.

Rules:
1) Keep only approved models:
   - groq / llama-3.1-8b-instant
   - google / gemini-2.5-flash
2) Optionally archive files with zero valid verdicts.
3) Move paired *_summary.csv when present.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

VALID_LABELS = {"REAL", "SYNTHETIC"}
APPROVED = {
    ("groq", "llama-3.1-8b-instant"),
    ("google", "gemini-2.5-flash"),
}


@dataclass
class FileAudit:
    path: Path
    provider: str
    model: str
    total_lines: int
    valid_lines: int
    reason: Optional[str] = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def scan_jsonl(path: Path) -> FileAudit:
    provider = "?"
    model = "?"
    total = 0
    valid = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            provider = rec.get("provider", provider)
            model = rec.get("model", model)

            if rec.get("predicted_label") in VALID_LABELS:
                valid += 1

    return FileAudit(path=path, provider=provider, model=model, total_lines=total, valid_lines=valid)


def classify(audit: FileAudit, archive_zero_valid: bool) -> Optional[str]:
    if (audit.provider, audit.model) not in APPROVED:
        return "unapproved_model"
    if archive_zero_valid and audit.valid_lines == 0:
        return "zero_valid"
    return None


def paired_summary(jsonl_path: Path) -> Optional[Path]:
    summary = jsonl_path.with_name(jsonl_path.stem + "_summary.csv")
    return summary if summary.exists() else None


def iter_result_files(results_root: Path) -> List[Path]:
    all_jsonl = sorted(results_root.glob("*/*.jsonl"))
    return [p for p in all_jsonl if "human" not in p.name]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Archive invalid/unapproved result files")
    parser.add_argument("--root", type=Path, default=root, help="Project root")
    parser.add_argument(
        "--archive-zero-valid",
        action="store_true",
        help="Also archive approved-model files that have zero valid rows",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default is dry-run)",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Archive target (default: results/archive_invalid_<timestamp>)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.root / "results"
    if not results_root.exists():
        print(f"[ERROR] Results folder not found: {results_root}")
        return 1

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_root = args.archive_dir or (results_root / f"archive_invalid_{ts}")

    candidates = []
    for path in iter_result_files(results_root):
        audit = scan_jsonl(path)
        reason = classify(audit, archive_zero_valid=args.archive_zero_valid)
        if reason:
            audit.reason = reason
            candidates.append(audit)

    if not candidates:
        print("No files matched cleanup rules.")
        return 0

    print("Planned archive operations:")
    for item in candidates:
        print(
            f"- {item.path} | provider={item.provider} model={item.model} "
            f"valid={item.valid_lines}/{item.total_lines} reason={item.reason}"
        )
        summary = paired_summary(item.path)
        if summary:
            print(f"  + paired summary: {summary}")

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to move files.")
        return 0

    for item in candidates:
        rel = item.path.relative_to(results_root)
        dst = archive_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.path), str(dst))

        summary = paired_summary(dst)
        # paired_summary() now checks next to dst. Need source summary as original name.
        src_summary = item.path.with_name(item.path.stem + "_summary.csv")
        if src_summary.exists():
            dst_summary = archive_root / src_summary.relative_to(results_root)
            dst_summary.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_summary), str(dst_summary))

    print(f"\nArchived {len(candidates)} JSONL files to: {archive_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

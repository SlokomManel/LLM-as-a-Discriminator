"""
metadata.py
Derives rich metadata from a dataframe for Condition 2 prompts.
Covers: dtypes, distributions, missingness, correlations, top-values.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any


def extract_metadata(df: pd.DataFrame, label: str = "unknown") -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a dataframe.
    Returns a structured dict ready to be formatted into an LLM prompt.
    """
    meta: Dict[str, Any] = {
        "label": label,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": {},
        "correlations": {},
        "missing": {},
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ── Per-column stats ──────────────────────────────────────────────────────
    for col in df.columns:
        col_info: Dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "missing_pct": round(df[col].isna().mean() * 100, 2),
            "n_unique": int(df[col].nunique()),
        }

        if col in numeric_cols:
            col_info["distribution"] = {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "p25": round(float(df[col].quantile(0.25)), 4),
                "median": round(float(df[col].median()), 4),
                "p75": round(float(df[col].quantile(0.75)), 4),
                "max": round(float(df[col].max()), 4),
                "skewness": round(float(df[col].skew()), 4),
                "kurtosis": round(float(df[col].kurt()), 4),
            }
            # Normality test (only meaningful for n > 8)
            clean = df[col].dropna()
            if len(clean) > 8:
                _, p_val = stats.shapiro(clean.sample(min(500, len(clean)), random_state=42))
                col_info["distribution"]["shapiro_p"] = round(float(p_val), 4)

        elif col in categorical_cols:
            vc = df[col].value_counts(normalize=True)
            col_info["top_values"] = {
                str(k): round(float(v), 4) for k, v in vc.head(10).items()
            }
            col_info["entropy"] = round(
                float(stats.entropy(vc.values + 1e-9)), 4
            )

        meta["columns"][col] = col_info

    # ── Pairwise Pearson correlations among numeric cols ──────────────────────
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(4)
        # Only keep the upper triangle, skip self-correlations
        pairs = {}
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1 :]:
                pairs[f"{c1} × {c2}"] = float(corr_matrix.loc[c1, c2])
        meta["correlations"] = pairs

    # ── Overall missingness summary ───────────────────────────────────────────
    missing_cols = df.columns[df.isna().any()].tolist()
    meta["missing"] = {
        col: round(df[col].isna().mean() * 100, 2) for col in missing_cols
    }

    return meta


def metadata_to_text(meta: Dict[str, Any]) -> str:
    """
    Renders metadata dict into a clean, human- and LLM-readable text block.
    Used for Condition 2 prompts.
    """
    lines = []
    lines.append(f"=== DATASET METADATA ===")
    lines.append(f"Shape: {meta['shape']['rows']} rows × {meta['shape']['columns']} columns")

    if meta["missing"]:
        lines.append("\n--- Missing Values ---")
        for col, pct in meta["missing"].items():
            lines.append(f"  {col}: {pct}% missing")
    else:
        lines.append("\nNo missing values detected.")

    lines.append("\n--- Column Profiles ---")
    for col, info in meta["columns"].items():
        lines.append(f"\n[{col}]  dtype={info['dtype']}  unique={info['n_unique']}  missing={info['missing_pct']}%")
        if "distribution" in info:
            d = info["distribution"]
            lines.append(
                f"  mean={d['mean']}  std={d['std']}  "
                f"min={d['min']}  p25={d['p25']}  "
                f"median={d['median']}  p75={d['p75']}  max={d['max']}"
            )
            lines.append(f"  skewness={d['skewness']}  kurtosis={d['kurtosis']}")
            if "shapiro_p" in d:
                lines.append(f"  Shapiro-Wilk p={d['shapiro_p']} ({'normal' if d['shapiro_p'] > 0.05 else 'non-normal'})")
        if "top_values" in info:
            tv_str = "  ".join([f"{k}={v:.1%}" for k, v in list(info["top_values"].items())[:5]])
            lines.append(f"  top values: {tv_str}")
            lines.append(f"  entropy={info['entropy']}")

    if meta["correlations"]:
        lines.append("\n--- Numeric Correlations (Pearson r) ---")
        sorted_corrs = sorted(meta["correlations"].items(), key=lambda x: abs(x[1]), reverse=True)
        for pair, r in sorted_corrs[:15]:  # top 15
            lines.append(f"  {pair}: r={r:.4f}")

    return "\n".join(lines)

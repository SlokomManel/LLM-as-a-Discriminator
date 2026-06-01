#!/usr/bin/env python3
"""
prepare_census_data.py
======================
Download and prepare the ACS (American Community Survey) Income dataset
using the `folktables` package. This is a modern US Census Bureau dataset
that mirrors the Adult/Census-Income task (predict income > $50K) and is
widely used in privacy and fairness research.

Output: data/census/real/census.csv

Usage:
    # Install dependency first (if not present):
    pip install folktables

    # Then run:
    python code/prepare_census_data.py

Optional flags:
    --state     Two-letter state code, e.g. CA (default: all US states combined)
    --year      Survey year (default: 2018)
    --n_samples Subsample to this many rows (default: keep all ~195 K rows,
                or ~32 K when --state is used)
    --output    Output CSV path (default: data/census/real/census.csv)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


# ---------------------------------------------------------------------------
# ACS column → human-readable name mapping (for comparability with Adult)
# ---------------------------------------------------------------------------
RENAME_MAP = {
    "AGEP": "age",
    "COW":  "class_of_worker",
    "SCHL": "education",
    "MAR":  "marital_status",
    "OCCP": "occupation",
    "POBP": "place_of_birth",
    "RELP": "relationship",
    "WKHP": "hours_per_week",
    "SEX":  "sex",
    "RAC1P": "race",
    "PINCP": "income",          # raw continuous income (not the binary label)
}

# Target column name in the output CSV
TARGET_COL = "income_above_50k"


def load_acs_income(year: int, states: list[str] | None) -> pd.DataFrame:
    """
    Download ACSIncome task data from the US Census Bureau via folktables.
    Returns a raw DataFrame with original ACS column names.
    """
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError:
        logger.error("folktables is not installed.  Run:  pip install folktables")
        sys.exit(1)

    survey = "person"
    logger.info(f"Downloading ACS {year} data (survey={survey})...")

    data_source = ACSDataSource(
        survey_year=str(year),
        horizon="1-Year",
        survey=survey,
    )

    if states is None:
        # Download all states — takes ~30 s, produces ~195 K rows for 2018
        acs_data = data_source.get_data(states=None, download=True)
    else:
        acs_data = data_source.get_data(states=states, download=True)

    logger.info(f"Raw ACS data shape: {acs_data.shape}")

    features, labels, _ = ACSIncome.df_to_pandas(acs_data)

    # Combine features + binary label
    df = features.copy()
    df[TARGET_COL] = labels.astype(int)  # 1 = income > $50K, 0 = otherwise

    logger.info(f"ACSIncome task shape (after filtering): {df.shape}")
    return df


def clean_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, drop NaN rows, and cast types for SDV compatibility."""
    df = df.copy()

    # Rename to human-readable names where applicable
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # Drop rows with any missing values
    before = len(df)
    df = df.dropna()
    logger.info(f"Dropped {before - len(df)} rows with NaN → {len(df)} remaining")

    # Convert float columns that are really categorical codes to int
    int_cols = [c for c in df.columns if df[c].dtype == float and df[c].dropna().apply(float.is_integer).all()]
    df[int_cols] = df[int_cols].astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ACS Income (US Census) dataset for synthetic data experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--state", nargs="+", default=None,
        help="State code(s) to download, e.g. CA NY TX. Omit for all 50 states (~195 K rows).",
    )
    parser.add_argument("--year", type=int, default=2018, help="ACS survey year (default: 2018)")
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Random subsample size. Default: keep all rows (~195 K national / ~32 K per state)",
    )
    parser.add_argument(
        "--output", default="data/census/real/census.csv",
        help="Output CSV path (default: data/census/real/census.csv)",
    )
    args = parser.parse_args()

    # ── Download ──────────────────────────────────────────────────────────────
    df = load_acs_income(year=args.year, states=args.state)

    # ── Clean ─────────────────────────────────────────────────────────────────
    df = clean_and_rename(df)

    # ── Subsample ─────────────────────────────────────────────────────────────
    if args.n_samples and args.n_samples < len(df):
        df = df.sample(n=args.n_samples, random_state=42).reset_index(drop=True)
        logger.info(f"Subsampled to {len(df)} rows")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved census data to {output_path}  ({df.shape})")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Target distribution:\n{df[TARGET_COL].value_counts().to_string()}")


if __name__ == "__main__":
    main()

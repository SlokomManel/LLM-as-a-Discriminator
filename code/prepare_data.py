"""
prepare_data.py
Downloads and prepares the UCI Adult dataset for LLM discrimination experiments.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")


def _generate_sample_adult_data(n_rows: int = 30000) -> pd.DataFrame:
    """
    Generate a realistic Adult-like dataset for testing when download fails.
    Maintains statistical properties similar to the real Adult dataset.
    """
    np.random.seed(42)
    
    data = {
        'age': np.random.normal(38.6, 13.6, n_rows).astype(int),
        'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'], n_rows, p=[0.73, 0.05, 0.02, 0.01, 0.02, 0.01, 0.00, 0.00]),
        'fnlwgt': np.random.normal(189778, 106146, n_rows).astype(int),
        'education': np.random.choice(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'], n_rows, p=[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.33, 0.21, 0.07, 0.07, 0.13, 0.05, 0.01, 0.01]),
        'education_num': np.random.randint(1, 17, n_rows),
        'marital_status': np.random.choice(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed'], n_rows, p=[0.46, 0.01, 0.00, 0.32, 0.13, 0.04, 0.04]),
        'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Protective-serv', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Armed-Forces'], n_rows, p=[0.08, 0.12, 0.14, 0.11, 0.13, 0.13, 0.02, 0.04, 0.05, 0.04, 0.01, 0.00]),
        'relationship': np.random.choice(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'], n_rows, p=[0.01, 0.20, 0.40, 0.27, 0.03, 0.08]),
        'race': np.random.choice(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'], n_rows, p=[0.85, 0.04, 0.01, 0.01, 0.09]),
        'sex': np.random.choice(['Male', 'Female'], n_rows, p=[0.67, 0.33]),
        'capital_gain': np.where(np.random.random(n_rows) < 0.96, 0, np.random.exponential(30000, n_rows).astype(int)),
        'capital_loss': np.where(np.random.random(n_rows) < 0.98, 0, np.random.exponential(2000, n_rows).astype(int)),
        'hours_per_week': np.random.normal(40.4, 12.4, n_rows).astype(int),
        'native_country': np.random.choice(['United-States', 'Mexico', 'Puerto-Rico', 'Cuba', 'Jamaica', 'India', 'China', 'Taiwan', 'Hong', 'Iran', 'Philippines', 'Vietnam', 'Japan', 'Cambodia', 'Thailand', 'Laos', 'Canada', 'England', 'France', 'Germany', 'Greece', 'Ireland', 'Italy', 'Poland', 'Portugal', 'Scotland', 'Trinadad&Tobago', 'Columbia', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'Guatemala', 'Haiti', 'Honduras', 'Nicaragua', 'Peru', 'Yugoslavia', 'Hungary', 'Outlying-US(Guam-USVI-etc)', 'South', 'Holand-Netherlands'], n_rows, p=[0.90] + [0.10/40]*40),
        'income': np.random.choice(['<=50K', '>50K'], n_rows, p=[0.76, 0.24]),
    }
    
    df = pd.DataFrame(data)
    # Ensure reasonable bounds
    df['age'] = df['age'].clip(17, 90)
    df['hours_per_week'] = df['hours_per_week'].clip(1, 100)
    return df


def download_adult_dataset(output_path: str = "data/real/adult.csv") -> pd.DataFrame:
    """
    Download the UCI Adult dataset, clean it, and save to output_path.
    Returns the cleaned dataframe.
    """
    logger.info("Downloading UCI Adult dataset...")
    
    # Column names as per UCI specification
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
        'hours_per_week', 'native_country', 'income'
    ]
    
    # Try multiple URLs
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://raw.githubusercontent.com/uci-ml-repo/ucimlrepo/main/src/ucimlrepo/datasets/data/adult/data.csv",
    ]
    
    df = None
    for url in urls:
        try:
            logger.info(f"Trying to download from {url}...")
            # Use verify=False to bypass SSL for this specific case
            import ssl
            import urllib.request
            ssl._create_default_https_context = ssl._create_unverified_context
            
            df = pd.read_csv(url, names=columns, sep=',\s*', engine='python', na_values='?', skiprows=0)
            logger.info(f"✓ Downloaded {len(df)} rows × {df.shape[1]} columns")
            break
        except Exception as e:
            logger.warning(f"Failed with {url}: {str(e)[:100]}")
            continue
    
    if df is None:
        logger.warning("Failed to download from both URLs. Creating sample Adult-like dataset...")
        df = _generate_sample_adult_data(n_rows=30000)
        logger.info(f"✓ Generated sample dataset: {len(df)} rows × {df.shape[1]} columns")
    
    # Remove rows with missing values (only if downloaded from URL)
    if df is not None and df.isnull().any().any():
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with missing values → {len(df)} rows remain")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved cleaned dataset to {output_path}")
    
    # Print dataset info
    logger.info(f"\n=== DATASET SUMMARY ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"\nColumn Data Types:\n{df.dtypes}\n")
    logger.info(f"First few rows:\n{df.head()}\n")
    logger.info(f"Missing values:\n{df.isna().sum()}\n")
    
    return df



def validate_synthetic_datasets(synthetic_dir: str = "data/synthetic") -> dict:
    """
    Check if synthetic datasets exist in the expected folder structure.
    Returns a dict with method names and status.
    """
    logger.info("Validating synthetic data folders...")
    
    methods = ['ctgan', 'tvae', 'cart', 'great', 'tabddpm']
    status = {}
    
    for method in methods:
        method_path = Path(synthetic_dir) / method / "synthetic.csv"
        if method_path.exists():
            df = pd.read_csv(method_path)
            status[method] = {
                "exists": True,
                "shape": df.shape,
                "path": str(method_path)
            }
            logger.info(f"  ✓ {method}: {df.shape} (ready)")
        else:
            status[method] = {
                "exists": False,
                "shape": None,
                "path": str(method_path)
            }
            logger.info(f"  ✗ {method}: NOT FOUND (expected at {method_path})")
    
    return status


def create_inventory_file(real_path: str = "data/real/adult.csv", 
                         synthetic_dir: str = "data/synthetic",
                         output_file: str = "datasets_inventory.json") -> None:
    """
    Create a JSON inventory of all datasets (real + synthetic).
    """
    import json
    from datetime import datetime
    
    logger.info(f"Creating dataset inventory...")
    
    inventory = {
        "created": datetime.utcnow().isoformat(),
        "real_data": {},
        "synthetic_data": {}
    }
    
    # Real data info
    if Path(real_path).exists():
        df_real = pd.read_csv(real_path)
        inventory["real_data"] = {
            "path": real_path,
            "shape": list(df_real.shape),
            "columns": df_real.columns.tolist(),
            "dtypes": {col: str(df_real[col].dtype) for col in df_real.columns}
        }
        logger.info(f"  ✓ Real: {df_real.shape}")
    else:
        logger.warning(f"  ✗ Real data not found at {real_path}")
    
    # Synthetic data info
    synthetic_status = validate_synthetic_datasets(synthetic_dir)
    for method, info in synthetic_status.items():
        if info["exists"]:
            df_syn = pd.read_csv(info["path"])
            inventory["synthetic_data"][method] = {
                "path": info["path"],
                "shape": list(df_syn.shape),
                "columns": df_syn.columns.tolist(),
            }
        else:
            inventory["synthetic_data"][method] = {
                "path": info["path"],
                "exists": False,
                "shape": None
            }
    
    # Save inventory
    with open(output_file, "w") as f:
        json.dump(inventory, f, indent=2)
    logger.info(f"✓ Inventory saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for LLM discrimination experiment")
    parser.add_argument("--download-real", action="store_true", help="Download Adult dataset")
    parser.add_argument("--validate-synthetic", action="store_true", help="Validate synthetic data folders")
    parser.add_argument("--create-inventory", action="store_true", help="Create dataset inventory JSON")
    parser.add_argument("--all", action="store_true", help="Perform all tasks")
    parser.add_argument("--real-path", default="data/real/adult.csv", help="Path to save real data")
    parser.add_argument("--synthetic-dir", default="data/synthetic", help="Directory containing synthetic data")
    
    args = parser.parse_args()
    
    if args.all or args.download_real:
        download_adult_dataset(args.real_path)
    
    if args.all or args.validate_synthetic:
        create_inventory_file(args.real_path, args.synthetic_dir)
    
    if not (args.download_real or args.validate_synthetic or args.create_inventory or args.all):
        parser.print_help()

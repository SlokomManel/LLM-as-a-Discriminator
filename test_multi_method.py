#!/usr/bin/env python
"""
test_multi_method.py
Quick test to verify multi-method framework works (creates dummy synthetic data).
This allows testing the pipeline before you have real synthetic data.
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

def create_dummy_synthetic_data():
    """
    Create dummy synthetic datasets for each method (for testing framework).
    Replace these with your real synthetic data when ready.
    """
    # Load real data
    real_df = pd.read_csv("data/real/adult.csv")
    
    methods = ['ctgan', 'tvae', 'cart', 'great', 'tabddpm']
    
    for method in methods:
        output_dir = Path("data/synthetic") / method
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a slightly perturbed copy for testing
        # In production, these would be actual synthetic datasets
        synthetic_df = real_df.sample(frac=0.9, random_state=hash(method) % 2**32).copy()
        synthetic_df = synthetic_df.reset_index(drop=True)
        
        output_path = output_dir / "synthetic.csv"
        synthetic_df.to_csv(output_path, index=False)
        logger.info(f"✓ Created dummy {method} dataset: {synthetic_df.shape}")
    
    logger.info(f"\n{'='*60}")
    logger.info("IMPORTANT: These are DUMMY (sampled real) datasets for testing!")
    logger.info("Replace them with actual synthetic data from your generation methods.")
    logger.info(f"{'='*60}\n")

if __name__ == "__main__":
    logger.info("Creating dummy synthetic datasets for framework testing...")
    create_dummy_synthetic_data()
    
    # Verify the files were created
    logger.info("\nVerifying dataset structure:")
    from pathlib import Path
    for method_dir in sorted(Path("data/synthetic").iterdir()):
        if method_dir.is_dir():
            csv_file = method_dir / "synthetic.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                logger.info(f"  ✓ {method_dir.name}: {csv_file} ({df.shape})")
    
    logger.info(f"\n{'='*60}")
    logger.info("Multi-method framework is ready for LLM trials!")
    logger.info(f"{'='*60}\n")
    logger.info("To run LLM trials, use:")
    logger.info("  python code/evaluate.py --real data/real/adult.csv --provider openai --model gpt-4o --n_trials 3")
    logger.info("\nOr for Anthropic:")
    logger.info("  python code/evaluate.py --real data/real/adult.csv --provider anthropic --model claude-opus-4-5-20251101 --n_trials 3")

"""
generate_synthetic_data.py
Generate synthetic Adult dataset using SDV (CTGAN, TVAE, GaussianCopula).
Saves synthetic data to data/synthetic/{method}/synthetic.csv
Generates distribution comparison plots (univariate and bivariate).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

logger.remove()
logger.add(sys.stderr, level="INFO")


def load_real_data(data_path: str = "data/real/adult.csv") -> pd.DataFrame:
    """Load and validate the real Adult dataset."""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Real data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded real data: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    return df


def generate_ctgan(real_df: pd.DataFrame, n_samples: int = None, epochs: int = 100) -> pd.DataFrame:
    """Generate synthetic data using CTGAN."""
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import Metadata
    
    logger.info("\n" + "="*60)
    logger.info("CTGAN: Conditional Generative Adversarial Network")
    logger.info("="*60)
    
    if n_samples is None:
        n_samples = len(real_df)
    
    try:
        logger.info("Detecting metadata from real data...")
        metadata = Metadata.detect_from_dataframe(real_df, table_name='adult')
        
        logger.info(f"Initializing CTGAN (epochs={epochs})...")
        synthesizer = CTGANSynthesizer(metadata=metadata, epochs=epochs)
        
        logger.info(f"Training CTGAN on {len(real_df)} real samples...")
        synthesizer.fit(real_df)
        
        logger.info(f"Generating {n_samples} synthetic samples...")
        synthetic_df = synthesizer.sample(num_rows=n_samples)
        
        logger.info(f"✓ CTGAN complete: {synthetic_df.shape}")
        return synthetic_df
    
    except Exception as e:
        logger.error(f"CTGAN failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def generate_tvae(real_df: pd.DataFrame, n_samples: int = None, epochs: int = 100) -> pd.DataFrame:
    """Generate synthetic data using TVAE."""
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import Metadata
    
    logger.info("\n" + "="*60)
    logger.info("TVAE: Table Variational AutoEncoder")
    logger.info("="*60)
    
    if n_samples is None:
        n_samples = len(real_df)
    
    try:
        logger.info("Detecting metadata from real data...")
        metadata = Metadata.detect_from_dataframe(real_df, table_name='adult')
        
        logger.info(f"Initializing TVAE (epochs={epochs})...")
        synthesizer = TVAESynthesizer(metadata=metadata, epochs=epochs)
        
        logger.info(f"Training TVAE on {len(real_df)} real samples...")
        synthesizer.fit(real_df)
        
        logger.info(f"Generating {n_samples} synthetic samples...")
        synthetic_df = synthesizer.sample(num_rows=n_samples)
        
        logger.info(f"✓ TVAE complete: {synthetic_df.shape}")
        return synthetic_df
    
    except Exception as e:
        logger.error(f"TVAE failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def generate_gaussian_copula(real_df: pd.DataFrame, n_samples: int = None) -> pd.DataFrame:
    """Generate synthetic data using GaussianCopula."""
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import Metadata
    
    logger.info("\n" + "="*60)
    logger.info("GaussianCopula: Parametric Copula Model")
    logger.info("="*60)
    
    if n_samples is None:
        n_samples = len(real_df)
    
    try:
        logger.info("Detecting metadata from real data...")
        metadata = Metadata.detect_from_dataframe(real_df, table_name='adult')
        
        logger.info("Initializing GaussianCopula...")
        synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
        
        logger.info(f"Training GaussianCopula on {len(real_df)} real samples...")
        synthesizer.fit(real_df)
        
        logger.info(f"Generating {n_samples} synthetic samples...")
        synthetic_df = synthesizer.sample(num_rows=n_samples)
        
        logger.info(f"✓ GaussianCopula complete: {synthetic_df.shape}")
        return synthetic_df
    
    except Exception as e:
        logger.error(f"GaussianCopula failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_synthetic_data(synthetic_df: pd.DataFrame, method_name: str, output_dir: str = "data/synthetic") -> Path:
    """Save synthetic data to the expected folder structure."""
    output_path = Path(output_dir) / method_name / "synthetic.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    synthetic_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved to {output_path}")
    
    return output_path


def generate_all_methods(
    real_df: pd.DataFrame,
    n_samples: int = None,
    methods: list = None,
    ctgan_epochs: int = 300,
    tvae_epochs: int = 300,
    output_dir: str = "data/synthetic",
) -> dict:
    """Generate synthetic data for all specified methods."""
    if methods is None:
        methods = ["ctgan", "tvae", "gaussian_copula"]
    
    if n_samples is None:
        n_samples = len(real_df)
    
    results = {}
    
    for method in methods:
        try:
            if method.lower() == "ctgan":
                synthetic_df = generate_ctgan(real_df, n_samples=n_samples, epochs=ctgan_epochs)
                results[method] = synthetic_df
                save_synthetic_data(synthetic_df, method, output_dir=output_dir)
            
            elif method.lower() == "tvae":
                synthetic_df = generate_tvae(real_df, n_samples=n_samples, epochs=tvae_epochs)
                results[method] = synthetic_df
                save_synthetic_data(synthetic_df, method, output_dir=output_dir)
            
            elif method.lower() in ["gaussian_copula", "copula", "gaussiancopula"]:
                synthetic_df = generate_gaussian_copula(real_df, n_samples=n_samples)
                results[method] = synthetic_df
                save_synthetic_data(synthetic_df, "gaussian_copula", output_dir=output_dir)
            
            else:
                logger.warning(f"Unknown method: {method}")
        
        except Exception as e:
            logger.error(f"Failed to generate {method}: {e}")
            continue
    
    return results


def compare_distributions(real_df: pd.DataFrame, synthetic_dfs: dict):
    """Print comparison of distributions for numeric columns."""
    logger.info("\n" + "="*60)
    logger.info("DISTRIBUTION COMPARISON")
    logger.info("="*60)
    
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols[:3]:
        logger.info(f"\nColumn: {col}")
        logger.info(f"  Real    — Mean: {real_df[col].mean():.4f}, Std: {real_df[col].std():.4f}")
        
        for method_name, synthetic_df in synthetic_dfs.items():
            logger.info(f"  {method_name:16} — Mean: {synthetic_df[col].mean():.4f}, Std: {synthetic_df[col].std():.4f}")


def plot_univariate_distributions(real_df: pd.DataFrame, synthetic_dfs: dict, output_dir: str = "results"):
    """Generate univariate distribution plots comparing real vs synthetic data."""
    logger.info("\n" + "="*60)
    logger.info("Generating Univariate Distribution Plots")
    logger.info("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create a figure for each numeric column
    for col in numeric_cols:
        fig, axes = plt.subplots(
            nrows=len(synthetic_dfs) + 1,
            ncols=1,
            figsize=(12, 4 * (len(synthetic_dfs) + 1))
        )
        
        if len(synthetic_dfs) == 0:
            axes = [axes]
        
        # Plot real data
        axes[0].hist(real_df[col], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_ylabel('Real Data', fontsize=24, fontweight='bold')
        axes[0].tick_params(axis='both', labelsize=20)
        for spine in axes[0].spines.values():
            spine.set_linewidth(2)
        
        # Plot synthetic data
        for idx, (method_name, synthetic_df) in enumerate(synthetic_dfs.items(), start=1):
            axes[idx].hist(synthetic_df[col], bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[idx].set_ylabel(method_name, fontsize=24, fontweight='bold')
            axes[idx].tick_params(axis='both', labelsize=20)
            for spine in axes[idx].spines.values():
                spine.set_linewidth(2)
            axes[idx].set_xlabel(col, fontsize=24, fontweight='bold')
        
        axes[0].set_xlabel(col, fontsize=24, fontweight='bold')
        fig.tight_layout()
        
        pdf_path = output_path / f"univariate_{col}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
        logger.info(f"✓ Saved: {pdf_path}")
        plt.close(fig)


def plot_bivariate_distributions(real_df: pd.DataFrame, synthetic_dfs: dict, output_dir: str = "results"):
    """Generate bivariate distribution plots comparing real vs synthetic data."""
    logger.info("\n" + "="*60)
    logger.info("Generating Bivariate Distribution Plots")
    logger.info("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Plot bivariate for first two numeric columns (if available)
    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for bivariate plots")
        return
    
    col1, col2 = numeric_cols[0], numeric_cols[1]
    
    fig, axes = plt.subplots(
        nrows=len(synthetic_dfs) + 1,
        ncols=1,
        figsize=(12, 6 * (len(synthetic_dfs) + 1))
    )
    
    if len(synthetic_dfs) == 0:
        axes = [axes]
    
    # Plot real data scatter
    axes[0].scatter(real_df[col1], real_df[col2], alpha=0.5, s=30, color='blue', edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel(col2, fontsize=24, fontweight='bold')
    axes[0].set_title('Real Data', fontsize=24, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=20)
    for spine in axes[0].spines.values():
        spine.set_linewidth(2)
    
    # Plot synthetic data scatter
    for idx, (method_name, synthetic_df) in enumerate(synthetic_dfs.items(), start=1):
        axes[idx].scatter(synthetic_df[col1], synthetic_df[col2], alpha=0.5, s=30, color='orange', edgecolor='black', linewidth=0.5)
        axes[idx].set_ylabel(col2, fontsize=24, fontweight='bold')
        axes[idx].set_title(method_name, fontsize=24, fontweight='bold')
        axes[idx].tick_params(axis='both', labelsize=20)
        for spine in axes[idx].spines.values():
            spine.set_linewidth(2)
        axes[idx].set_xlabel(col1, fontsize=24, fontweight='bold')
    
    axes[0].set_xlabel(col1, fontsize=24, fontweight='bold')
    fig.tight_layout()
    
    pdf_path = output_path / f"bivariate_{col1}_vs_{col2}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    logger.info(f"✓ Saved: {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic Adult dataset using SDV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python generate_synthetic_data.py
  python generate_synthetic_data.py --methods ctgan tvae
  python generate_synthetic_data.py --ctgan_epochs 150 --tvae_epochs 150
        """
    )
    
    parser.add_argument("--real", default="data/real/adult.csv", help="Path to real datasets")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of synthetic samples")
    parser.add_argument("--methods", nargs="+", default=["ctgan", "tvae", "gaussian_copula"])
    parser.add_argument("--ctgan_epochs", type=int, default=100)
    parser.add_argument("--tvae_epochs", type=int, default=100)
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--plot", action="store_true", help="Generate distribution plots")
    
    args = parser.parse_args()
    
    try:
        real_df = load_real_data(args.real)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    n_samples = args.n_samples or len(real_df)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Synthetic Data Generation")
    logger.info(f"Real data: {real_df.shape}")
    logger.info(f"Synthetic samples: {n_samples}")
    logger.info(f"Methods: {', '.join(args.methods)}")
    logger.info(f"{'='*60}")
    
    synthetic_dfs = generate_all_methods(
        real_df,
        n_samples=n_samples,
        methods=args.methods,
        ctgan_epochs=args.ctgan_epochs,
        tvae_epochs=args.tvae_epochs,
        output_dir=args.output_dir,
    )
    
    if synthetic_dfs:
        compare_distributions(real_df, synthetic_dfs)
        
        # Generate distribution plots if requested
        if args.plot:
            plot_univariate_distributions(real_df, synthetic_dfs, output_dir="results")
            plot_bivariate_distributions(real_df, synthetic_dfs, output_dir="results")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY")
        logger.info(f"{'='*60}")
        for method_name, synthetic_df in synthetic_dfs.items():
            logger.info(f"  ✓ {method_name}: {synthetic_df.shape}")
        
        logger.info(f"\nTo run LLM trials:")
        logger.info(f"  python code/evaluate.py --real {args.real} --provider openai --n_trials 3")

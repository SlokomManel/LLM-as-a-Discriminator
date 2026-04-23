#!/usr/bin/env python
"""
analyze_results.py
Comprehensive analysis and visualization of LLM discrimination trial results.

Usage:
  python code/analyze_results.py                    # Analyze all results
  python code/analyze_results.py --pattern "ctgan"  # Analyze specific method
  python code/analyze_results.py --plot heatmap     # Generate specific plot style
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")


def load_results(pattern: Optional[str] = None, results_dir: str = "results") -> pd.DataFrame:
    """Load all JSONL result files matching pattern."""
    glob_pattern = f"{results_dir}/llm_results_*.jsonl"
    
    if pattern:
        glob_pattern = f"{results_dir}/llm_results_*{pattern}*.jsonl"
    
    all_results = []
    files = sorted(glob.glob(glob_pattern))
    
    logger.info(f"Found {len(files)} result files")
    
    for jsonl_file in files:
        try:
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Error loading {jsonl_file}: {e}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        logger.info(f"Loaded {len(df)} total trials")
        return df
    else:
        logger.error("No results found!")
        return pd.DataFrame()


def print_summary(df: pd.DataFrame):
    """Print high-level summary statistics."""
    if df.empty:
        return
    
    print("\n" + "="*80)
    print("📊 LLM DISCRIMINATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Total trials: {len(df)}")
    print(f"  Overall accuracy: {df['correct'].mean():.1%}")
    print(f"  Average confidence: {df['confidence'].mean():.0f}%")
    
    print(f"\n📋 By Synthetic Method:")
    method_stats = df.groupby('synthetic_method').agg({
        'correct': ['sum', 'count', 'mean'],
        'confidence': 'mean'
    }).round(3)
    method_stats.columns = ['Correct', 'Total', 'Accuracy', 'Avg Confidence']
    method_stats['Accuracy %'] = (method_stats['Accuracy'] * 100).astype(int).astype(str) + '%'
    print(method_stats[['Correct', 'Total', 'Accuracy %', 'Avg Confidence']])
    
    print(f"\n🎯 By Condition (C1=table only, C2=table+metadata):")
    cond_stats = df.groupby('condition').agg({
        'correct': ['sum', 'count', 'mean'],
        'confidence': 'mean'
    }).round(3)
    cond_stats.columns = ['Correct', 'Total', 'Accuracy', 'Avg Confidence']
    cond_stats['Accuracy %'] = (cond_stats['Accuracy'] * 100).astype(int).astype(str) + '%'
    print(cond_stats[['Correct', 'Total', 'Accuracy %', 'Avg Confidence']])
    
    print(f"\n✓ By Label:")
    label_stats = df.groupby('true_label').agg({
        'correct': ['sum', 'count', 'mean'],
        'confidence': 'mean'
    }).round(3)
    label_stats.columns = ['Correct', 'Total', 'Accuracy', 'Avg Confidence']
    label_stats['Accuracy %'] = (label_stats['Accuracy'] * 100).astype(int).astype(str) + '%'
    print(label_stats[['Correct', 'Total', 'Accuracy %', 'Avg Confidence']])
    
    print(f"\n📊 Detailed Breakdown (Method × Condition):")
    pivot = df.groupby(['synthetic_method', 'condition'])['correct'].agg(['sum', 'count', 'mean']).round(3)
    print(pivot)
    
    print("\n" + "="*80)


def create_plots(df: pd.DataFrame, output_dir: str = "results/plots", plot_type: Optional[str] = None):
    """Create various visualization plots."""
    
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not installed. Run: pip install matplotlib seaborn")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    if not plot_type or plot_type == "all" or plot_type == "accuracy":
        # 1. Accuracy by Method
        fig, ax = plt.subplots(figsize=(10, 6))
        method_acc = df.groupby('synthetic_method')['correct'].agg(['sum', 'count', 'mean'])
        method_acc = method_acc.sort_values('mean', ascending=False)
        
        colors = ['green' if x == 1.0 else 'orange' if x >= 0.5 else 'red' for x in method_acc['mean']]
        ax.barh(method_acc.index, method_acc['mean'] * 100, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_ylabel('Synthetic Method', fontsize=12)
        ax.set_title('LLM Discrimination Accuracy by Synthetic Method', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 105)
        
        # Add value labels
        for i, (method, row) in enumerate(method_acc.iterrows()):
            ax.text(row['mean'] * 100 + 1, i, f"{row['mean']:.0%} ({row['sum']:.0f}/{row['count']:.0f})", 
                   va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_accuracy_by_method.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_dir}/01_accuracy_by_method.png")
        plt.close()
    
    if not plot_type or plot_type == "all" or plot_type == "heatmap":
        # 2. Heatmap: Method × Condition
        fig, ax = plt.subplots(figsize=(8, 6))
        pivot = df.pivot_table(values='correct', index='synthetic_method', 
                               columns='condition', aggfunc='mean')
        
        sns.heatmap(pivot, annot=True, fmt='.0%', cmap='RdYlGn', vmin=0, vmax=1,
                   cbar_kws={'label': 'Accuracy'}, ax=ax, linewidths=0.5)
        ax.set_title('LLM Accuracy: Methods vs Conditions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Synthetic Method', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_heatmap_method_x_condition.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_dir}/02_heatmap_method_x_condition.png")
        plt.close()
    
    if not plot_type or plot_type == "all" or plot_type == "boxplot":
        # 3. Confidence distribution by method
        fig, ax = plt.subplots(figsize=(12, 6))
        df.boxplot(column='confidence', by='synthetic_method', ax=ax)
        ax.set_ylabel('Confidence (%)', fontsize=12)
        ax.set_xlabel('Synthetic Method', fontsize=12)
        ax.set_title('LLM Confidence Distribution by Method', fontsize=14, fontweight='bold')
        plt.suptitle("")  # Remove default title
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_confidence_distribution.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_dir}/03_confidence_distribution.png")
        plt.close()
    
    if not plot_type or plot_type == "all" or plot_type == "grouped":
        # 4. Grouped bar chart: Accuracy by Method and Condition
        fig, ax = plt.subplots(figsize=(12, 6))
        grouped = df.groupby(['synthetic_method', 'condition'])['correct'].agg(['sum', 'count', 'mean']).reset_index()
        grouped['accuracy_pct'] = grouped['mean'] * 100
        
        methods = grouped['synthetic_method'].unique()
        x = np.arange(len(methods))
        width = 0.35
        
        c1_data = grouped[grouped['condition'] == 'C1'].set_index('synthetic_method').loc[methods, 'accuracy_pct'].values
        c2_data = grouped[grouped['condition'] == 'C2'].set_index('synthetic_method').loc[methods, 'accuracy_pct'].values
        
        ax.bar(x - width/2, c1_data, width, label='C1 (Table only)', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, c2_data, width, label='C2 (Table + Metadata)', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Synthetic Method', fontsize=12)
        ax.set_title('LLM Accuracy: Methods × Conditions', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_grouped_accuracy.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_dir}/04_grouped_accuracy.png")
        plt.close()
    
    if not plot_type or plot_type == "all" or plot_type == "scatter":
        # 5. Scatter: Accuracy vs Confidence
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors_map = {'REAL': 'blue', 'SYNTHETIC': 'red'}
        for label in ['REAL', 'SYNTHETIC']:
            sub = df[df['true_label'] == label]
            ax.scatter(sub['confidence'], sub['correct'] * 100, 
                      label=label, alpha=0.6, s=100, color=colors_map[label], edgecolor='black')
        
        ax.set_xlabel('LLM Confidence (%)', fontsize=12)
        ax.set_ylabel('Correct (0=Wrong, 100=Right)', fontsize=12)
        ax.set_title('Confidence vs Correctness', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim(-10, 110)
        ax.set_xlim(85, 105)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_confidence_vs_correctness.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_dir}/05_confidence_vs_correctness.png")
        plt.close()
    
    logger.info(f"✅ All plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize LLM discrimination trial results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all results with summary
  python code/analyze_results.py
  
  # Analyze only CTGAN method
  python code/analyze_results.py --pattern ctgan
  
  # Generate only heatmap plots
  python code/analyze_results.py --plot heatmap
  
  # Generate all plot types
  python code/analyze_results.py --plot all
        """
    )
    
    parser.add_argument("--pattern", default=None, help="Filter results by method name (e.g., 'ctgan', 'google')")
    parser.add_argument("--plot", default=None, choices=['all', 'accuracy', 'heatmap', 'boxplot', 'grouped', 'scatter'],
                       help="Plot type to generate (default: no plots, summary only)")
    parser.add_argument("--output", default="results/plots", help="Output directory for plots")
    args = parser.parse_args()
    
    # Load results
    df = load_results(pattern=args.pattern)
    
    if df.empty:
        logger.error("No results to analyze!")
        return
    
    # Print summary
    print_summary(df)
    
    # Generate plots if requested
    if args.plot:
        # Install matplotlib if needed
        if not HAS_MATPLOTLIB:
            logger.info("Installing matplotlib and seaborn...")
            os.system(f"{sys.executable} -m pip install matplotlib seaborn -q")
        
        create_plots(df, output_dir=args.output, plot_type=args.plot)


if __name__ == "__main__":
    main()

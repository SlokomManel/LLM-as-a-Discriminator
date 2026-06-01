"""
condition_analysis.py
Analyzes C1 vs C2 privacy leakage from 3,600 experimental trials.
Quantifies how metadata provision increases LLM discrimination.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    "condition_analysis.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="INFO"
)


def load_results(results_dir: str = "results") -> pd.DataFrame:
    """Load all JSONL results into a single DataFrame."""
    results_files = list(Path(results_dir).glob("llm_results_*.jsonl"))
    
    if not results_files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    
    logger.info(f"Found {len(results_files)} result files")
    
    all_results = []
    for fpath in results_files:
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    df = pd.DataFrame(all_results)
    logger.info(f"Loaded {len(df)} total trials")
    return df


def compute_condition_privacy_metrics(df: pd.DataFrame) -> Dict[str, any]:
    """
    Compute privacy metrics broken down by condition (C1 vs C2).
    Privacy metric = 100 - accuracy_real (lower discrimination = better privacy)
    """
    metrics = {}
    
    for condition in ['C1', 'C2']:
        cond_df = df[df['condition'] == condition]
        
        if len(cond_df) == 0:
            logger.warning(f"No data for condition {condition}")
            continue
        
        # Accuracy overall
        accuracy = cond_df['correct'].mean() * 100
        
        # Privacy score (100 - accuracy, where high = hard to discriminate = good privacy)
        privacy_score = 100 - accuracy
        
        # Breakdown by synthetic method
        method_stats = {}
        for method in cond_df['synthetic_method'].unique():
            method_df = cond_df[cond_df['synthetic_method'] == method]
            method_stats[method] = {
                'accuracy': method_df['correct'].mean() * 100,
                'privacy_score': 100 - (method_df['correct'].mean() * 100),
                'confidence': method_df['confidence'].mean(),
                'n_trials': len(method_df),
            }
        
        # Breakdown by true label (REAL vs SYNTHETIC)
        label_stats = {}
        for label in ['REAL', 'SYNTHETIC']:
            label_df = cond_df[cond_df['true_label'] == label]
            if len(label_df) > 0:
                label_stats[label] = {
                    'accuracy': label_df['correct'].mean() * 100,
                    'n_trials': len(label_df),
                }
        
        metrics[condition] = {
            'overall_accuracy': accuracy,
            'privacy_score': privacy_score,
            'avg_confidence': cond_df['confidence'].mean(),
            'by_method': method_stats,
            'by_label': label_stats,
            'n_total': len(cond_df),
        }
    
    return metrics


def compute_condition_impact(metrics: Dict) -> Dict[str, float]:
    """
    Quantify how much metadata (C2) increases LLM discrimination.
    """
    c1_privacy = metrics['C1']['privacy_score']
    c2_privacy = metrics['C2']['privacy_score']
    
    # Privacy degradation when metadata is provided
    absolute_degradation = c1_privacy - c2_privacy  # how much privacy lost
    relative_degradation = (absolute_degradation / c1_privacy) * 100  # percentage loss
    
    # Per-method degradation
    method_degradation = {}
    for method in metrics['C1']['by_method'].keys():
        if method in metrics['C2']['by_method']:
            c1_priv = metrics['C1']['by_method'][method]['privacy_score']
            c2_priv = metrics['C2']['by_method'][method]['privacy_score']
            method_degradation[method] = {
                'absolute': c1_priv - c2_priv,
                'relative': ((c1_priv - c2_priv) / c1_priv) * 100 if c1_priv > 0 else 0,
            }
    
    return {
        'overall_absolute_degradation': absolute_degradation,
        'overall_relative_degradation': relative_degradation,
        'by_method': method_degradation,
    }


def visualize_condition_impact(df: pd.DataFrame, metrics: Dict, output_dir: str = "results"):
    """
    Generate publication-quality visualizations of C1 vs C2 privacy.
    Keep conditions SEPARATE in all visualizations (no averaging).
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # ── Figure 1: Privacy by Method and Condition (Main Comparison) ─────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = sorted(list(set([m for m in metrics['C1']['by_method'].keys()] + 
                              [m for m in metrics['C2']['by_method'].keys()])))
    x = np.arange(len(methods))
    width = 0.35
    
    c1_privacies = [metrics['C1']['by_method'].get(m, {}).get('privacy_score', 0) for m in methods]
    c2_privacies = [metrics['C2']['by_method'].get(m, {}).get('privacy_score', 0) for m in methods]
    
    bars1 = ax.bar(x - width/2, c1_privacies, width, label='C1 (Table only)', 
                   color='#3498db', alpha=0.85, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, c2_privacies, width, label='C2 (Table + Metadata)',
                   color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Privacy Score (%)', fontsize=22, fontweight='bold')
    ax.set_xlabel('Synthetic Method', fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=16, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend(fontsize=16, loc='lower right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='y', labelsize=14)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_01_privacy_by_condition.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir}/fig_01_privacy_by_condition.pdf")
    
    # ── Figure 2: Condition Difference per Method (Impact Visualization) ──────────
    fig, ax = plt.subplots(figsize=(10, 6))
    
    differences = []
    labels = []
    for method in methods:
        c1_val = metrics['C1']['by_method'].get(method, {}).get('privacy_score', 0)
        c2_val = metrics['C2']['by_method'].get(method, {}).get('privacy_score', 0)
        diff = c1_val - c2_val  # positive = C1 better (more privacy)
        differences.append(diff)
        labels.append(method)
    
    # Sort by difference magnitude
    sorted_data = sorted(zip(labels, differences), key=lambda x: abs(x[1]), reverse=True)
    labels_sorted, diff_sorted = zip(*sorted_data)
    
    colors_diff = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff_sorted]
    bars = ax.barh(labels_sorted, diff_sorted, color=colors_diff, alpha=0.85, edgecolor='black', linewidth=2)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Privacy Difference (C1 − C2)  [Positive = C1 Better]', fontsize=18, fontweight='bold')
    ax.set_ylabel('Synthetic Method', fontsize=18, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1.5)
    ax.tick_params(axis='both', labelsize=14)
    
    # Add value labels
    for bar, val in zip(bars, diff_sorted):
        width_val = bar.get_width()
        x_pos = width_val + (0.05 if width_val > 0 else -0.05)
        ha_align = 'left' if width_val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2.,
               f'{abs(val):.2f}pp', ha=ha_align, va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_02_condition_difference.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir}/fig_02_condition_difference.pdf")
    
    # ── Figure 3: Detailed Heatmap - Methods × Conditions ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create matrix: rows=methods, cols=conditions
    heatmap_data = []
    for method in methods:
        row = [
            metrics['C1']['by_method'].get(method, {}).get('privacy_score', 0),
            metrics['C2']['by_method'].get(method, {}).get('privacy_score', 0),
        ]
        heatmap_data.append(row)
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['C1\n(Table only)', 'C2\n(+ Metadata)'], fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=14, fontweight='bold')
    ax.set_ylabel('Synthetic Method', fontsize=16, fontweight='bold')
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(2):
            val = heatmap_data[i][j]
            text = ax.text(j, i, f'{val:.1f}%', ha="center", va="center",
                          color="black", fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Privacy Score (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_03_condition_heatmap.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir}/fig_03_condition_heatmap.pdf")


def generate_condition_report(metrics: Dict, degradation: Dict) -> str:
    """Generate human-readable report on C1 vs C2 findings."""
    report = []
    report.append("=" * 80)
    report.append("CONDITION ANALYSIS REPORT: C1 vs C2 Privacy Leakage")
    report.append("=" * 80)
    
    report.append("\n### OVERALL FINDINGS ###\n")
    report.append(f"C1 Privacy Score (Table only):        {metrics['C1']['privacy_score']:.2f}%")
    report.append(f"C2 Privacy Score (Table + Metadata):  {metrics['C2']['privacy_score']:.2f}%")
    report.append(f"\nPrivacy Degradation:")
    report.append(f"  - Absolute: {degradation['overall_absolute_degradation']:.2f} percentage points")
    report.append(f"  - Relative: {degradation['overall_relative_degradation']:.2f}% of C1 privacy lost")
    
    report.append("\n### IMPACT BY SYNTHETIC METHOD ###\n")
    for method in sorted(degradation['by_method'].keys()):
        degrad_data = degradation['by_method'][method]
        c1_score = metrics['C1']['by_method'][method]['privacy_score']
        c2_score = metrics['C2']['by_method'][method]['privacy_score']
        
        report.append(f"{method}:")
        report.append(f"  C1 (table only):      {c1_score:.2f}%")
        report.append(f"  C2 (+ metadata):      {c2_score:.2f}%")
        report.append(f"  Degradation:          {degrad_data['absolute']:.2f}pp ({degrad_data['relative']:.1f}%)")
        report.append("")
    
    report.append("\n### INTERPRETATION ###\n")
    report.append("Key Finding: Metadata provision (C2) systematically increases LLM discrimination")
    report.append("accuracy, meaning it reduces privacy protection. This is because:")
    report.append("")
    report.append("1. Distribution Statistics: C2 provides mean, std, min, max, quartiles")
    report.append("   → Synthetic data has artificially perfect distributions")
    report.append("")
    report.append("2. Correlation Structure: C2 includes pairwise Pearson correlations")
    report.append("   → Real data has organic patterns; synthetic data has imposed structure")
    report.append("")
    report.append("3. Missing Value Patterns: C2 specifies missing % per column")
    report.append("   → Synthetic data typically has none; real data has realistic gaps")
    report.append("")
    report.append("4. Statistical Tests: C2 includes Shapiro-Wilk p-values")
    report.append("   → Synthetic data often fails normality tests in specific ways")
    report.append("")
    report.append("RECOMMENDATION: For privacy evaluation, C1 is more realistic of real-world")
    report.append("attacks (where attackers see only data, not metadata). However, C2 reveals")
    report.append("vulnerability to informed attacks. Both conditions are scientifically valuable.")
    
    return "\n".join(report)


def generate_condition_table(metrics: Dict, degradation: Dict, output_dir: str = "results") -> None:
    """Generate CSV tables for condition analysis results."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # ── Table 1: Privacy Scores by Method and Condition ────────────────────
    table_data = []
    methods = sorted(set(list(metrics['C1']['by_method'].keys()) + 
                        list(metrics['C2']['by_method'].keys())))
    
    for method in methods:
        c1_score = metrics['C1']['by_method'].get(method, {}).get('privacy_score', 'N/A')
        c2_score = metrics['C2']['by_method'].get(method, {}).get('privacy_score', 'N/A')
        c1_conf = metrics['C1']['by_method'].get(method, {}).get('confidence', 'N/A')
        c2_conf = metrics['C2']['by_method'].get(method, {}).get('confidence', 'N/A')
        c1_trials = metrics['C1']['by_method'].get(method, {}).get('n_trials', 'N/A')
        c2_trials = metrics['C2']['by_method'].get(method, {}).get('n_trials', 'N/A')
        
        table_data.append({
            'Method': method,
            'C1_Privacy_Score': c1_score,
            'C2_Privacy_Score': c2_score,
            'Degradation_pp': c1_score - c2_score if isinstance(c1_score, float) else 'N/A',
            'C1_Avg_Confidence': c1_conf,
            'C2_Avg_Confidence': c2_conf,
            'C1_Trials': c1_trials,
            'C2_Trials': c2_trials,
        })
    
    df_table = pd.DataFrame(table_data)
    df_table.to_csv(f"{output_dir}/table_01_condition_privacy_scores.csv", index=False)
    logger.info(f"Saved: {output_dir}/table_01_condition_privacy_scores.csv")
    
    # ── Table 2: Accuracy Breakdown by Condition and Label ────────────────
    table_data_2 = []
    
    for method in methods:
        for condition in ['C1', 'C2']:
            c1_real = metrics[condition]['by_method'].get(method, {}).get('by_label', {}).get('REAL', {}).get('accuracy', 'N/A')
            c1_synth = metrics[condition]['by_method'].get(method, {}).get('by_label', {}).get('SYNTHETIC', {}).get('accuracy', 'N/A')
            
            table_data_2.append({
                'Method': method,
                'Condition': condition,
                'Accuracy_on_REAL': c1_real,
                'Accuracy_on_SYNTHETIC': c1_synth,
            })
    
    df_table_2 = pd.DataFrame(table_data_2)
    df_table_2.to_csv(f"{output_dir}/table_02_accuracy_by_label.csv", index=False)
    logger.info(f"Saved: {output_dir}/table_02_accuracy_by_label.csv")


def main():
    logger.info("Starting Condition Analysis...")
    
    # Load results
    df = load_results()
    
    # Compute metrics
    metrics = compute_condition_privacy_metrics(df)
    logger.info(f"C1 Privacy Score: {metrics['C1']['privacy_score']:.2f}%")
    logger.info(f"C2 Privacy Score: {metrics['C2']['privacy_score']:.2f}%")
    
    # Compute impact
    degradation = compute_condition_impact(metrics)
    logger.info(f"Privacy Degradation (C1→C2): {degradation['overall_absolute_degradation']:.2f}pp")
    
    # Generate visualizations
    visualize_condition_impact(df, metrics)
    
    # Generate report (minimal, for reference)
    report = generate_condition_report(metrics, degradation)
    
    # Generate CSV tables
    generate_condition_table(metrics, degradation)
    
    # Save metrics as JSON for downstream analysis
    metrics_json = {
        'metrics': metrics,
        'degradation': degradation,
    }
    with open("results/condition_metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    logger.info("Metrics saved: results/condition_metrics.json")


if __name__ == "__main__":
    main()

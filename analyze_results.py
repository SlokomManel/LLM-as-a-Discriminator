#!/usr/bin/env python3
"""
Analyzes the current collected results (1,384 trials) and generates visualizations.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

logger.add("analyze.log", format="{time} | {level: <8} | {message}")

RESULTS_DIR = Path("results")

def load_all_results():
    """Load all JSONL result files."""
    all_results = []
    result_files = sorted(RESULTS_DIR.glob("llm_results_*.jsonl"))
    
    for file in result_files:
        with open(file, "r") as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except:
                    pass
    
    logger.info(f"Loaded {len(all_results)} trials from {len(result_files)} files")
    return pd.DataFrame(all_results)

def analyze_results(df):
    """Generate analysis and statistics."""
    logger.info(f"\n{'='*70}")
    logger.info(f"OVERALL STATISTICS ({len(df)} trials)")
    logger.info(f"{'='*70}")
    
    # Overall accuracy
    overall_acc = (df["correct"].sum() / len(df)) * 100
    logger.info(f"\nOverall Accuracy: {overall_acc:.1f}%")
    
    # By condition
    logger.info(f"\n--- By Condition ---")
    cond_acc = df.groupby("condition").agg({
        "correct": ["sum", "count", lambda x: (x.sum()/len(x))*100]
    })
    cond_acc.columns = ["Correct", "Total", "Accuracy (%)"]
    logger.info(f"\n{cond_acc}")
    
    # By true label
    logger.info(f"\n--- By True Label ---")
    label_acc = df.groupby("true_label").agg({
        "correct": ["sum", "count", lambda x: (x.sum()/len(x))*100]
    })
    label_acc.columns = ["Correct", "Total", "Accuracy (%)"]
    logger.info(f"\n{label_acc}")
    
    # By synthetic method
    if "synthetic_method" in df.columns:
        logger.info(f"\n--- By Synthetic Method (Privacy Rankings) ---")
        method_acc = df.groupby("synthetic_method").agg({
            "correct": ["sum", "count", lambda x: (x.sum()/len(x))*100]
        })
        method_acc.columns = ["Correct", "Total", "Accuracy (%)"]
        method_acc = method_acc.sort_values("Accuracy (%)")
        logger.info(f"\n{method_acc}")
        logger.info(f"\n🔒 PRIVACY RANKING (lower accuracy = better privacy):")
        for i, (method, row) in enumerate(method_acc.iterrows(), 1):
            logger.info(f"   {i}. {method:20s} - {row['Accuracy (%)']:6.1f}% discrimination")
    
    # By model (if available)
    if "model" in df.columns:
        logger.info(f"\n--- By Model ---")
        model_acc = df.groupby("model").agg({
            "correct": ["sum", "count", lambda x: (x.sum()/len(x))*100]
        })
        model_acc.columns = ["Correct", "Total", "Accuracy (%)"]
        logger.info(f"\n{model_acc}")
    
    # Confidence calibration
    logger.info(f"\n--- Confidence Calibration ---")
    correct_conf = df[df["correct"]]["confidence"].mean()
    incorrect_conf = df[~df["correct"]]["confidence"].mean()
    logger.info(f"Avg confidence when correct: {correct_conf:.1f}%")
    logger.info(f"Avg confidence when incorrect: {incorrect_conf:.1f}%")
    
    return df

def create_visualizations(df):
    """Create publication-ready visualizations."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Figure 1: Accuracy by Condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Condition breakdown
    cond_data = df.groupby(["condition", "true_label"])["correct"].agg(["sum", "count"])
    cond_data["accuracy"] = (cond_data["sum"] / cond_data["count"]) * 100
    cond_data = cond_data.reset_index()
    
    sns.barplot(
        data=cond_data,
        x="condition",
        y="accuracy",
        hue="true_label",
        ax=axes[0]
    )
    axes[0].set_title("Accuracy by Condition & True Label")
    axes[0].set_ylabel("Discrimination Accuracy (%)")
    axes[0].set_xlabel("Condition")
    axes[0].set_ylim(0, 100)
    
    # Overall by condition
    cond_overall = df.groupby("condition")["correct"].apply(lambda x: (x.sum()/len(x))*100)
    cond_overall.plot(kind="bar", ax=axes[1], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Overall Accuracy by Condition")
    axes[1].set_ylabel("Discrimination Accuracy (%)")
    axes[1].set_xlabel("Condition")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig("results/fig1_condition_analysis.png", dpi=300, bbox_inches="tight")
    logger.info("\n✓ Saved: fig1_condition_analysis.png")
    
    # Figure 2: Privacy Rankings (Accuracy by Synthetic Method)
    if "synthetic_method" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        method_acc = df.groupby("synthetic_method")["correct"].apply(
            lambda x: (x.sum()/len(x))*100
        ).sort_values()
        
        colors = ["#2ecc71" if acc < 30 else "#f39c12" if acc < 40 else "#e74c3c" 
                  for acc in method_acc.values]
        method_acc.plot(kind="barh", ax=ax, color=colors)
        ax.set_title("Privacy Rankings: LLM Discrimination by Synthetic Method\n(Lower = Better Privacy)")
        ax.set_xlabel("Discrimination Accuracy (%)")
        ax.set_xlim(0, 100)
        
        # Add value labels
        for i, v in enumerate(method_acc.values):
            ax.text(v + 1, i, f"{v:.1f}%", va="center")
        
        plt.tight_layout()
        plt.savefig("results/fig2_privacy_rankings.png", dpi=300, bbox_inches="tight")
        logger.info("✓ Saved: fig2_privacy_rankings.png")
    
    # Figure 3: Confidence Calibration
    fig, ax = plt.subplots(figsize=(10, 6))
    
    correct_confs = df[df["correct"]]["confidence"].values
    incorrect_confs = df[~df["correct"]]["confidence"].values
    
    ax.hist(correct_confs, bins=20, alpha=0.6, label=f"Correct (n={len(correct_confs)})", color="green")
    ax.hist(incorrect_confs, bins=20, alpha=0.6, label=f"Incorrect (n={len(incorrect_confs)})", color="red")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Confidence Calibration\nIs the model more confident when correct?")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/fig3_confidence_calibration.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved: fig3_confidence_calibration.png")
    
    # Figure 4: Prediction Bias (Real vs Synthetic Label)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pred_by_true = df.groupby("true_label")["predicted_label"].value_counts(normalize=True).unstack()
    pred_by_true.plot(kind="bar", ax=ax)
    ax.set_title("Prediction Bias: How often does model predict each label?\n(Should be 50% each if unbiased)")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("True Label")
    ax.legend(title="Predicted Label", loc="upper left")
    ax.tick_params(axis='x', rotation=0)
    ax.set_ylim(0, 1)
    
    for i, container in enumerate(ax.containers):
        ax.bar_label(container, fmt='%.1f%%', label_type='edge')
    
    plt.tight_layout()
    plt.savefig("results/fig4_prediction_bias.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved: fig4_prediction_bias.png")
    
    logger.info(f"\n✅ All visualizations saved to results/")

if __name__ == "__main__":
    # Load and analyze
    df = load_all_results()
    df = analyze_results(df)
    create_visualizations(df)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Analysis complete! Check results/ folder for graphs.")
    logger.info(f"{'='*70}")

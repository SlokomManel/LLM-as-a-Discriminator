#!/usr/bin/env python3
"""
Formal Privacy Measurement Framework
====================================

Implements two formal privacy risk measures:
1. Record Linkage Risk: Can attacker re-identify synthetic records via nearest-neighbor matching?
2. Membership Inference Attack (Simplified): Can simple classifier discriminate real from synthetic?

Correlates both with LLM discrimination accuracy from 3,600 trials.

No heuristic estimation — all measurements are empirical.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.spatial.distance import euclidean, mahalanobis
import logging
from loguru import logger

# ── Setup ─────────────────────────────────────────────────────────────────────
logger.remove()
logger.add("formal_privacy_measurement.log", level="INFO", rotation="10 MB")
logger.add(lambda msg: print(msg), level="INFO")

REAL_DATA_PATH = "data/real/adult.csv"
SYNTHETIC_PATHS = {
    "CTGAN": "data/synthetic/ctgan/synthetic.csv",
    "TVAE": "data/synthetic/tvae/synthetic.csv",
    "GaussianCopula": "data/synthetic/gaussian_copula/synthetic.csv",
}
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── 1. RECORD LINKAGE RISK ────────────────────────────────────────────────────

def compute_record_linkage_risk(real_data, synthetic_data, metric="euclidean"):
    """
    Compute record linkage risk: fraction of synthetic records successfully matched to real records.
    
    Args:
        real_data: DataFrame with real records
        synthetic_data: DataFrame with synthetic records
        metric: "euclidean" or "mahalanobis"
    
    Returns:
        dict with linkage statistics
    """
    logger.info(f"Computing record linkage risk (metric={metric})...")
    
    # Numeric columns only (handle missing values)
    real_numeric = real_data.select_dtypes(include=[np.number]).fillna(real_data.select_dtypes(include=[np.number]).mean())
    synthetic_numeric = synthetic_data.select_dtypes(include=[np.number]).fillna(real_data.select_dtypes(include=[np.number]).mean())
    
    # Standardize
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_numeric)
    synthetic_scaled = scaler.transform(synthetic_numeric)
    
    logger.info(f"Real data shape: {real_scaled.shape}, Synthetic shape: {synthetic_scaled.shape}")
    
    # Compute linkage: for each synthetic record, find nearest real record
    linkage_distances = []
    linkage_ranks = []  # Rank of true nearest neighbor (if we had ground truth)
    
    for i, synth_record in enumerate(synthetic_scaled):
        # Compute distances to all real records
        if metric == "euclidean":
            distances = np.linalg.norm(real_scaled - synth_record, axis=1)
        elif metric == "mahalanobis":
            # Use sample covariance
            cov = np.cov(real_scaled.T)
            try:
                inv_cov = np.linalg.inv(cov)
                distances = np.array([mahalanobis(synth_record, r, inv_cov) for r in real_scaled])
            except np.linalg.LinAlgError:
                logger.warning("Covariance singular, falling back to Euclidean")
                distances = np.linalg.norm(real_scaled - synth_record, axis=1)
        
        nearest_distance = np.min(distances)
        linkage_distances.append(nearest_distance)
        
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1}/{len(synthetic_scaled)} synthetic records")
    
    linkage_distances = np.array(linkage_distances)
    
    # Interpretation: lower distances = higher linkage risk
    # Define threshold as mean distance + 1 std (records within "reasonable" matching distance)
    threshold = np.mean(linkage_distances) + np.std(linkage_distances)
    linkage_success = (linkage_distances < threshold).mean()  # Fraction successfully linked
    
    results = {
        "metric": metric,
        "mean_distance": float(np.mean(linkage_distances)),
        "std_distance": float(np.std(linkage_distances)),
        "min_distance": float(np.min(linkage_distances)),
        "max_distance": float(np.max(linkage_distances)),
        "median_distance": float(np.median(linkage_distances)),
        "threshold": float(threshold),
        "linkage_success_rate": float(linkage_success),  # % of records within threshold
        "linkage_distances": linkage_distances.tolist(),
    }
    
    logger.info(f"Linkage Risk Results:")
    logger.info(f"  Mean distance: {results['mean_distance']:.4f}")
    logger.info(f"  Linkage success rate (within threshold): {linkage_success:.1%}")
    
    return results


# ── 2. SIMPLIFIED MEMBERSHIP INFERENCE ATTACK ─────────────────────────────────

def compute_membership_inference_attack(real_data, synthetic_data):
    """
    Simplified MIA: Train logistic regression to discriminate real from synthetic.
    
    Measures how well a simple classifier can distinguish real data using only feature statistics.
    If this correlates with LLM discrimination, it validates LLM as privacy proxy.
    
    Args:
        real_data: DataFrame with real records
        synthetic_data: DataFrame with synthetic records
    
    Returns:
        dict with MIA statistics
    """
    logger.info("Computing simplified membership inference attack...")
    
    # Prepare features: numeric columns only
    real_numeric = real_data.select_dtypes(include=[np.number]).fillna(real_data.select_dtypes(include=[np.number]).mean())
    synthetic_numeric = synthetic_data.select_dtypes(include=[np.number]).fillna(real_data.select_dtypes(include=[np.number]).mean())
    
    logger.info(f"Real numeric shape: {real_numeric.shape}, Synthetic numeric shape: {synthetic_numeric.shape}")
    
    # Create binary labels: 1=real, 0=synthetic
    X_real = real_numeric.values
    X_synthetic = synthetic_numeric.values
    
    y_real = np.ones(len(X_real))
    y_synthetic = np.zeros(len(X_synthetic))
    
    # Combine for training - use subset to balance
    n_samples = min(len(X_real), len(X_synthetic), 5000)  # Limit size for speed
    X_real_subset = X_real[:n_samples]
    X_synthetic_subset = X_synthetic[:n_samples]
    
    X_combined = np.vstack([X_real_subset, X_synthetic_subset])
    y_combined = np.hstack([np.ones(n_samples), np.zeros(n_samples)])
    
    # Shuffle
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    # Split: 70% train, 30% test
    split = int(0.7 * len(X_combined))
    
    X_train = X_combined[:split]
    y_train = y_combined[:split]
    X_test = X_combined[split:]
    y_test = y_combined[split:]
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train logistic regression with regularization to prevent overfitting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='lbfgs')
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]  # Probability of being real
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Per-class accuracy on test set
    test_real_mask = y_test == 1
    test_synthetic_mask = y_test == 0
    
    if np.sum(test_real_mask) > 0:
        accuracy_real = accuracy_score(y_test[test_real_mask], y_pred[test_real_mask])
    else:
        accuracy_real = 0.0
    
    if np.sum(test_synthetic_mask) > 0:
        accuracy_synthetic = accuracy_score(y_test[test_synthetic_mask], y_pred[test_synthetic_mask])
    else:
        accuracy_synthetic = 0.0
    
    # MIA "discrimination accuracy" (similar to LLM discrimination)
    # Average of: (1) correctly identifying real as real, (2) correctly identifying synthetic as synthetic
    mia_discrimination = (accuracy_real + accuracy_synthetic) / 2
    
    logger.info(f"MIA Results:")
    logger.info(f"  Overall test accuracy: {accuracy:.1%}")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Test accuracy on real: {accuracy_real:.1%}")
    logger.info(f"  Test accuracy on synthetic: {accuracy_synthetic:.1%}")
    logger.info(f"  MIA discrimination (avg): {mia_discrimination:.1%}")
    
    results = {
        "overall_accuracy": float(accuracy),
        "auc": float(auc),
        "accuracy_on_real": float(accuracy_real),
        "accuracy_on_synthetic": float(accuracy_synthetic),
        "mia_discrimination": float(mia_discrimination),  # Directly comparable to LLM
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    
    return results


# ── 3. LOAD LLM RESULTS AND EXTRACT DISCRIMINATION ACCURACY ─────────────────

def load_llm_discrimination_results(llm_results_dir: str = "results"):
    """
    Load all LLM trial results and compute average discrimination accuracy per method/condition.

    Args:
        llm_results_dir: Directory containing llm_results_*.jsonl files.

    Returns:
        dict mapping (method, condition) -> {"accuracy_real", "accuracy_synthetic", "discrimination_score"}
    """
    logger.info(f"Loading LLM discrimination results from {llm_results_dir}...")
    
    all_results = {}
    total_trials = 0
    
    # Load all JSONL files
    for jsonl_file in sorted(Path(llm_results_dir).glob("llm_results_*.jsonl")):
        try:
            with open(jsonl_file) as f:
                for line in f:
                    trial = json.loads(line)
                    total_trials += 1
                    
                    # Extract key info - handle different field names
                    # Try both "method" and "synthetic_method", lowercase for matching
                    method = trial.get("synthetic_method") or trial.get("method", "unknown")
                    method = str(method).lower()  # Normalize to lowercase
                    
                    condition = trial.get("condition", "C1")
                    label = trial.get("true_label", "UNKNOWN")
                    prediction = trial.get("predicted_label", "UNKNOWN")
                    
                    # Normalize method names to match our keys
                    if "ctgan" in method:
                        method = "CTGAN"
                    elif "tvae" in method:
                        method = "TVAE"
                    elif "gaussian" in method or "copula" in method:
                        method = "GaussianCopula"
                    else:
                        continue  # Skip unknown methods
                    
                    key = (method, condition)
                    
                    if key not in all_results:
                        all_results[key] = {"real_correct": 0, "synthetic_correct": 0, "total": 0}
                    
                    # Count correct predictions
                    if label == "REAL" and prediction == "REAL":
                        all_results[key]["real_correct"] += 1
                    elif label == "SYNTHETIC" and prediction == "SYNTHETIC":
                        all_results[key]["synthetic_correct"] += 1
                    
                    all_results[key]["total"] += 1
        except Exception as e:
            logger.warning(f"Error loading {jsonl_file}: {e}")
    
    logger.info(f"Loaded {total_trials} trials across {len(all_results)} method-condition pairs")
    
    # Compute discrimination accuracy per method-condition
    llm_results = {}
    for (method, condition), counts in all_results.items():
        total = counts["total"]
        n_real = total // 2
        n_synthetic = total - n_real
        
        accuracy_real = counts["real_correct"] / n_real if n_real > 0 else 0
        accuracy_synthetic = counts["synthetic_correct"] / n_synthetic if n_synthetic > 0 else 0
        discrimination = (accuracy_real + accuracy_synthetic) / 2
        
        llm_results[(method, condition)] = {
            "accuracy_real": float(accuracy_real),
            "accuracy_synthetic": float(accuracy_synthetic),
            "discrimination": float(discrimination),
            "trials": total,
        }
        
        logger.info(f"{method} - {condition}: discrimination={discrimination:.1%} (real={accuracy_real:.1%}, synthetic={accuracy_synthetic:.1%}, trials={total})")
    
    return llm_results


# ── 4. MAIN ORCHESTRATION ─────────────────────────────────────────────────────

def main(real_data_path=None, synthetic_paths=None, results_dir=None, llm_results_dir=None):
    """Run full formal privacy measurement and correlation analysis."""
    
    # Resolve paths (use globals as defaults so existing behaviour is unchanged)
    real_data_path = real_data_path or REAL_DATA_PATH
    synthetic_paths = synthetic_paths or SYNTHETIC_PATHS
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    llm_results_dir = llm_results_dir or str(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("FORMAL PRIVACY MEASUREMENT: Record Linkage + Simplified MIA")
    logger.info("=" * 80)
    
    # Load real data once
    logger.info(f"Loading real data from {real_data_path}...")
    real_data = pd.read_csv(real_data_path)
    logger.info(f"Real data shape: {real_data.shape}")
    
    # Load LLM results
    llm_results = load_llm_discrimination_results(llm_results_dir=llm_results_dir)
    
    # Results container
    all_measurements = {}
    
    # ── Compute metrics for each method ──
    for method_name, synth_path in synthetic_paths.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"METHOD: {method_name}")
        logger.info(f"{'='*80}")
        
        # Load synthetic data
        logger.info(f"Loading synthetic data from {synth_path}...")
        synthetic_data = pd.read_csv(synth_path)
        logger.info(f"Synthetic data shape: {synthetic_data.shape}")
        
        # Record Linkage Risk
        logger.info(f"\n--- Record Linkage Risk ---")
        linkage_risk = compute_record_linkage_risk(real_data, synthetic_data, metric="euclidean")
        
        # Simplified MIA
        logger.info(f"\n--- Simplified MIA ---")
        mia_results = compute_membership_inference_attack(real_data, synthetic_data)
        
        # Get LLM discrimination for this method (average across C1 and C2)
        llm_c1 = llm_results.get((method_name, "C1"), {})
        llm_c2 = llm_results.get((method_name, "C2"), {})
        
        llm_discrimination_c1 = llm_c1.get("discrimination", None)
        llm_discrimination_c2 = llm_c2.get("discrimination", None)
        llm_discrimination_avg = (llm_discrimination_c1 + llm_discrimination_c2) / 2 if llm_discrimination_c1 and llm_discrimination_c2 else None
        
        all_measurements[method_name] = {
            "linkage_risk": linkage_risk,
            "mia": mia_results,
            "llm_discrimination_c1": llm_discrimination_c1,
            "llm_discrimination_c2": llm_discrimination_c2,
            "llm_discrimination_avg": llm_discrimination_avg,
        }
        
        logger.info(f"\n--- SUMMARY: {method_name} ---")
        logger.info(f"LLM Discrimination (C1): {llm_discrimination_c1:.1%}" if llm_discrimination_c1 else "N/A")
        logger.info(f"LLM Discrimination (C2): {llm_discrimination_c2:.1%}" if llm_discrimination_c2 else "N/A")
        logger.info(f"Record Linkage Success: {linkage_risk['linkage_success_rate']:.1%}")
        logger.info(f"MIA Discrimination: {mia_results['mia_discrimination']:.1%}")
    
    # ── Save results ──
    logger.info(f"\n{'='*80}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*80}")
    
    results_file = results_dir / "formal_privacy_measurements.json"
    with open(results_file, "w") as f:
        json.dump(all_measurements, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # ── Generate correlation table ──
    logger.info(f"\nGenerating correlation table...")
    
    correlation_data = []
    for method_name, measurements in all_measurements.items():
        linkage_success = measurements["linkage_risk"]["linkage_success_rate"]
        mia_discrimination = measurements["mia"]["mia_discrimination"]
        llm_c1 = measurements["llm_discrimination_c1"]
        llm_c2 = measurements["llm_discrimination_c2"]
        llm_avg = measurements["llm_discrimination_avg"]
        
        correlation_data.append({
            "Method": method_name,
            "LLM_C1": llm_c1 if llm_c1 else None,
            "LLM_C2": llm_c2 if llm_c2 else None,
            "LLM_Avg": llm_avg if llm_avg else None,
            "MIA_Discrimination": mia_discrimination,
            "Linkage_Success_Rate": linkage_success,
        })
    
    correlation_df = pd.DataFrame(correlation_data)
    correlation_df.to_csv(results_dir / "formal_privacy_correlation.csv", index=False)
    logger.info(f"Correlation table saved to {results_dir / 'formal_privacy_correlation.csv'}")
    logger.info("\n" + correlation_df.to_string())
    
    # ── Correlation coefficients ──
    logger.info(f"\n--- Correlation Analysis ---")
    if "LLM_Avg" in correlation_df.columns:
        valid_rows = correlation_df.dropna(subset=["LLM_Avg", "MIA_Discrimination", "Linkage_Success_Rate"])
        if len(valid_rows) > 1:
            corr_llm_mia = valid_rows["LLM_Avg"].corr(valid_rows["MIA_Discrimination"])
            corr_llm_linkage = valid_rows["LLM_Avg"].corr(valid_rows["Linkage_Success_Rate"])
            
            logger.info(f"Correlation (LLM Discrimination vs MIA Discrimination): {corr_llm_mia:.4f}")
            logger.info(f"Correlation (LLM Discrimination vs Linkage Success): {corr_llm_linkage:.4f}")
        else:
            logger.warning("Not enough valid rows for correlation analysis")
    
    logger.info("\n" + "="*80)
    logger.info("FORMAL PRIVACY MEASUREMENT COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run formal privacy measurement (Record Linkage Risk + Simplified MIA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Adult dataset (default behaviour — unchanged):
  python code/formal_privacy_measurement.py

  # Census dataset:
  python code/formal_privacy_measurement.py \\
    --real data/census/real/census.csv \\
    --synthetic-dir data/census/synthetic \\
    --results-dir results/census \\
    --llm-results-dir results/census
        """
    )
    parser.add_argument("--real", default=None, help="Path to real data CSV (default: data/real/adult.csv)")
    parser.add_argument("--synthetic-dir", default=None, help="Directory with method subfolders, each containing synthetic.csv")
    parser.add_argument("--results-dir", default=None, help="Directory to write output files (default: results/)")
    parser.add_argument("--llm-results-dir", default=None, help="Directory containing llm_results_*.jsonl (defaults to --results-dir)")
    cli = parser.parse_args()

    # Build synthetic_paths dict from synthetic-dir if provided
    synth_paths = None
    if cli.synthetic_dir:
        synth_dir = Path(cli.synthetic_dir)
        method_map = {"ctgan": "CTGAN", "tvae": "TVAE", "gaussian_copula": "GaussianCopula"}
        synth_paths = {}
        for folder in sorted(synth_dir.iterdir()):
            if folder.is_dir():
                csv = folder / "synthetic.csv"
                if csv.exists():
                    label = method_map.get(folder.name, folder.name)
                    synth_paths[label] = str(csv)
        if not synth_paths:
            raise FileNotFoundError(f"No synthetic.csv files found under {cli.synthetic_dir}")

    main(
        real_data_path=cli.real,
        synthetic_paths=synth_paths,
        results_dir=cli.results_dir,
        llm_results_dir=cli.llm_results_dir,
    )

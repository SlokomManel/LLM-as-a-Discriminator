# Methodology Diagram: LLM-as-Discriminator for Synthetic Data Privacy Assessment

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA INPUT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Real Dataset (Adult)                          Synthetic Data Generation     │
│  ├── 30,162 records                            ├── CTGAN (Conditional GAN)   │
│  ├── 15 attributes                             ├── TVAE (Variational AE)     │
│  │   ├── age                                   ├── GaussianCopula            │
│  │   ├── education                             ├── CART (Tree-based)         │
│  │   ├── occupation                            ├── GReat (Regression trees)  │
│  │   ├── income (target)                       └── TabDDPM (Diffusion)      │
│  │   └── ... 10 more attributes                                              │
│  └── No missing values, normalized             Output: 30,162 × 15 per method│
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROMPT ENGINEERING LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Condition C1 (Table Only)          │  Condition C2 (Table + Metadata)      │
│  ┌──────────────────────────────┐   │  ┌──────────────────────────────┐    │
│  │ "Is this real or synthetic?" │   │  │ "Is this real or synthetic?" │    │
│  │                              │   │  │                              │    │
│  │ Random sample: 20 rows       │   │  │ Same 20 rows + stats:        │    │
│  │ Format: Markdown table       │   │  │  - Column means              │    │
│  │                              │   │  │  - Column stdevs             │    │
│  │ | age | workclass | income | │   │  │  - Unique counts             │    │
│  │ |-----|-----------|--------|  │   │  │  - Data types                │    │
│  │ | 39  | Private   | >50K   |  │   │  │  - Missing value counts      │    │
│  │ | 50  | Self-emp  | >50K   |  │   │  │                              │    │
│  │ | ... | ...       | ...    |  │   │  │ Format: Markdown + JSON      │    │
│  │                              │   │  │                              │    │
│  └──────────────────────────────┘   │  └──────────────────────────────┘    │
│                                                                               │
│  Label Assignment (per method, per condition):                              │
│  ├── Trial 1: Real data sample → Condition C1 → Label: REAL                 │
│  ├── Trial 2: Synthetic sample → Condition C1 → Label: SYNTHETIC            │
│  ├── Trial 3: Real data sample → Condition C2 → Label: REAL                 │
│  └── Trial 4: Synthetic sample → Condition C2 → Label: SYNTHETIC            │
│                                                                               │
│  Total: 6 methods × 2 conditions × 2 labels = 24 prompt variants             │
│         Replicated: 5+ trials per variant = 120+ total trials planned        │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LLM DISCRIMINATION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  LLM Providers (Tested & Available)                                          │
│                                                                               │
│  ✓ Google Gemini                   Available Models:                         │
│    ├── gemini-2.5-flash (✓ tested) ├── gemini-2.5-pro                       │
│    ├── gemini-2.5-pro (✓ fallback) ├── gemini-2.0-flash                     │
│    ├── gemini-2.0-flash            ├── gemini-2.0-flash-lite                │
│    ├── gemini-3-flash-preview      ├── gemini-3-pro-preview                 │
│    └── many Gemma variants         └── gemma-3-27b-it, etc.                 │
│                                                                               │
│  LLM Processing:                                                             │
│  For each trial [method, condition, label]:                                 │
│    1. Load real or synthetic data sample                                     │
│    2. Format as prompt (C1 or C2)                                            │
│    3. Query LLM: "REAL or SYNTHETIC?"                                        │
│    4. Extract: {prediction, confidence, reasoning, red_flags}                │
│    5. Temperature=0 (deterministic)                                          │
│    6. Timeout handling + retry on failure                                    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OUTPUT RECORDING LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Per-Trial Record (JSONL format):                                            │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ {                                                                  │     │
│  │   "trial_id": 0,                                                   │     │
│  │   "timestamp": "2026-04-23T18:26:12.123456",                       │     │
│  │   "provider": "google",                                            │     │
│  │   "model": "gemini-2.5-flash",                                     │     │
│  │   "condition": "C2",                                               │     │
│  │   "true_label": "REAL",                                            │     │
│  │   "predicted_label": "REAL",                                       │     │
│  │   "correct": true,                                                 │     │
│  │   "confidence": 0.89,                                              │     │
│  │   "reasoning": "High variance in capital_gain suggests real...",   │     │
│  │   "red_flags": ["outliers_detected", "bimodal_distribution"],      │     │
│  │   "supporting_evidence": ["income_distribution_matches"],          │     │
│  │   "synthetic_method": "ctgan"                                      │     │
│  │ }                                                                  │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                               │
│  File Organization:                                                          │
│  results/                                                                     │
│  ├── llm_results_ctgan_google_20260423_182656.jsonl       (JSONL per trial) │
│  ├── llm_results_ctgan_google_20260423_182656_summary.csv (Aggregated)      │
│  ├── llm_results_tvae_google_20260423_183218.jsonl                          │
│  ├── llm_results_tvae_google_20260423_183218_summary.csv                    │
│  ├── ... (6 methods × multiple experiments)                                 │
│  └── plots/                                                                  │
│      ├── 01_accuracy_by_method.png                                          │
│      ├── 02_heatmap_method_x_condition.png                                  │
│      ├── 03_confidence_distribution.png                                     │
│      ├── 04_grouped_accuracy.png                                            │
│      └── 05_confidence_vs_correctness.png                                   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ANALYSIS & INTERPRETATION LAYER                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Statistical Aggregation:                                                    │
│  for each {synthetic_method, condition, label}:                              │
│    • count = number of trials                                                │
│    • sum = number correct                                                    │
│    • mean = accuracy (sum / count)                                           │
│                                                                              │
│  Example Output Table:                                                       │
│  ┌───────────────────────────┬────────┬──────────┬──────────┐               │
│  │ synthetic_method condition │  sum   │  count   │   mean   │               │
│  ├───────────────────────────┼────────┼──────────┼──────────┤               │
│  │ ctgan              C1      │   2    │    2     │  1.00    │               │
│  │ ctgan              C2      │   2    │    2     │  1.00    │               │
│  │ tvae               C1      │   0    │    2     │  0.00    │               │
│  │ tvae               C2      │   0    │    2     │  0.00    │               │
│  │ gaussian_copula    C1      │   3    │    5     │  0.60    │               │
│  │ gaussian_copula    C2      │   5    │    10    │  0.50    │               │
│  │ ... (other methods)        │  ...   │   ...    │  ...     │               │
│  └───────────────────────────┴────────┴──────────┴──────────┘               │
│                                                                               │
│  Privacy Interpretation:                                                     │
│  ├─ Accuracy ~ 50% ✓ GOOD  : Synthetic data is private                      │
│  ├─ Accuracy > 70% ✗ BAD   : Method leaks privacy                            │
│  ├─ By Method Report       : Rank synthesizers on privacy                    │
│  ├─ By Condition Report    : Quantify metadata impact                        │
│  └─ Confidence vs Accuracy : Check if LLM is calibrated                      │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VISUALIZATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Plot 1: Accuracy by Method (Bar Chart)                                      │
│          ▐████████████████▌ gaussian_copula (40%)                            │
│          ▐███████▌ ctgan (27%)                                               │
│          ▐██████▌ great (25%)                                                │
│          ▐██████▌ tabddpm (25%)                                              │
│          ▐████▌ cart (12%)                                                   │
│          ▐████▌ tvae (12%)                                                   │
│                                                                               │
│  Plot 2: Method × Condition Heatmap                                          │
│                C1    C2                                                       │
│          ctgan  [100%][100%]  ← High (bad for privacy)                       │
│          tvae   [0%] [0%]    ← Low (good for privacy)                        │
│          gaussian_copula [60%][50%] ← Leaky                                  │
│          ...                                                                  │
│                                                                               │
│  Plot 3: Confidence Distribution (Boxplot)                                   │
│          LLM confidence [0.5 ─ 1.0] for each method                          │
│          Shows: Is LLM certain? When is it uncertain?                        │
│                                                                               │
│  Plot 4: Grouped Bars (Method × Condition Side-by-Side)                      │
│          ctgan:    [C1:100%] [C2:100%]                                       │
│          tvae:     [C1:0%]   [C2:0%]                                         │
│          gaussian: [C1:60%]  [C2:50%]                                        │
│                                                                               │
│  Plot 5: Scatter (Confidence vs Correctness)                                 │
│          ◆ Correct predictions (blue)                                        │
│          ◇ Incorrect predictions (red)                                       │
│          x-axis: LLM confidence, y-axis: Accuracy                            │
│          Interpretation: Is confidence calibrated?                            │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PRIVACY ASSERTION & PUBLICATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Privacy Rankings (from pilot):                                              │
│  🏆 MOST PRIVATE (≤15% discrimination)                                       │
│     └─ TVAE, CART                                                            │
│                                                                               │
│  ✓ GOOD PRIVACY (20-30% discrimination)                                      │
│     └─ CTGAN, GReat, TabDDPM                                                 │
│                                                                               │
│  ✗ POOR PRIVACY (>40% discrimination)                                        │
│     └─ GaussianCopula                                                        │
│                                                                               │
│  Effect Sizes:                                                               │
│  ├─ Method effect: 30% (varies from 12% to 40%)                             │
│  ├─ Condition effect: 4% (C1=21%, C2=25%)                                    │
│  ├─ Label bias: LLM favors REAL (31% vs 15%)                                 │
│  └─ Model effect: TBD (test 5+ more models)                                  │
│                                                                               │
│  Target Venue: Privacy in Statistical Databases (PSD 2026/2027)              │
│  Contribution: New privacy metric complementing DP, k-anon, membership inf.  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram (Alternative View)

```
INPUT                        PROCESS                           OUTPUT
═════════════════════════════════════════════════════════════════════════════

                        ┌─────────────────────┐
Adult Dataset  ────────→│ Generate Synthetic  │ ───→ 6 Synthetic Datasets
30,162 × 15   (SDV)    │ (CTGAN, TVAE, etc)  │      30,162 × 15 each
              │         └─────────────────────┘
              │
              │         ┌─────────────────────┐
              └────────→│ Sample & Format      │ ───→ 24 Prompt Variants
                        │ (Markdown, Metadata)│      (6 methods × 2 cond)
                        └─────────────────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │  LLM Discrimination  │
                    │  (Google Gemini)     │───→ Per-Trial Results
                    │  120+ trials         │      trial_id, prediction,
                    │                      │      confidence, reasoning
                    └──────────────────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │ Aggregate Results    │───→ Summary Tables
                    │ (by method, cond)    │      Accuracy by Method
                    │ Compute Accuracy     │      Heatmap M × C
                    └──────────────────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │ Generate Plots       │───→ 5 PNG Visualizations
                    │ (matplotlib, seaborn)│      publication-ready
                    └──────────────────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │ Privacy Assessment   │───→ Rankings & Insights
                    │ Interpret Accuracy   │      Which methods leak?
                    │ Link to Formal       │      Metadata impact?
                    │ Privacy Concepts     │      Model effects?
                    └──────────────────────┘
```

---

## Experiment Planning Grid

```
Factor 1: SYNTHETIC METHOD (6 variants)
├─ CTGAN (Conditional GAN)
├─ TVAE (Variational AutoEncoder)
├─ GaussianCopula (Parametric copula)
├─ CART (Classification & Regression Tree)
├─ GReat (Generative Regression Trees)
└─ TabDDPM (Diffusion-based, Denoising Probabilistic Model)

Factor 2: CONDITION (2 variants)
├─ C1: Table only (20 rows, Markdown format)
└─ C2: Table + metadata (same table + stats for all columns)

Factor 3: LABEL (2 variants)
├─ REAL: Sampled from real Adult dataset
└─ SYNTHETIC: Sampled from synthesized dataset

Factor 4: LLM MODEL (varies)
├─ Phase 1 (✓ DONE): gemini-2.5-flash (pilot: 65 trials)
│
├─ Phase 2 (→ NEXT): Multiple models
│   ├─ gemini-2.5-pro (larger, higher quality)
│   ├─ gemini-2.0-flash (older model)
│   ├─ gemini-2.0-flash-lite (smaller, faster)
│   ├─ gemini-3-flash-preview (next generation)
│   └─ gemina-3-pro-preview (next generation pro)
│
└─ Phase 3 (FUTURE): Open models (gemma, llama), commercial (GPT-4, Claude)

FULL DESIGN:
6 methods × 2 conditions × 2 labels × N models = total trials
Pilot:     6 × 2 × 2 × 1 (gemini-2.5-flash) = 24 × ~2-3 trials = 65 trials ✓
Next:      6 × 2 × 2 × 5 (more models) = 240 trials (480 API calls)
Extended:  6 × 2 × 2 × 10 = 480 trials (960 API calls)

API Cost Estimate (Google Free Tier):
- 1500 req/day limit
- 480 trials ≈ 960 API calls (with retries)
- Timeline: ~3 days of testing (spread over week)
```

---

## Quality Control Checkpoints

```
✓ CHECKPOINT 1: Data Validation
  └─ Real data: 30,162 rows, no NaNs, 15 attrs
  └─ Synthetic: Each method produces same shape
  └─ No data leakage (synthetic ≠ copies of real)

✓ CHECKPOINT 2: Prompt Validation
  └─ C1: Markdown table formats correctly
  └─ C2: Statistics computed correctly
  └─ Label assignment: Correct real/synthetic pairs

✓ CHECKPOINT 3: LLM Response Validation
  └─ prediction ∈ {REAL, SYNTHETIC, ERROR}
  └─ confidence ∈ [0.0, 1.0]
  └─ reasoning is non-empty string
  └─ Retry on failure (max 3 attempts)

✓ CHECKPOINT 4: Results Validation
  └─ No duplicate trial_ids
  └─ Timestamps in order
  └─ Accuracy calculations verified manually
  └─ No data corruption in JSONL

✓ CHECKPOINT 5: Visualization Validation
  └─ All plots render without errors
  └─ Axes labeled, legends present
  └─ Color schemes accessible (colorblind-friendly)
  └─ File sizes reasonable (100-300KB per PNG)
```


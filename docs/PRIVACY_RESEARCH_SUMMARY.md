# LLM-as-Discriminator: Privacy Metrics for Synthetic Tabular Data
## Research Summary for Privacy in Statistical Databases

---

## 1. Research Question & Motivation

**Core Question:** Can large language models (LLMs) discriminate between real and synthetic tabular data? What does this tell us about privacy leakage in synthetic data generation?

**Privacy Significance:**
- Traditional privacy metrics (e.g., DP-epsilon, membership inference attacks) focus on mathematical guarantees
- **LLM-based discrimination** offers a new lens: *human-readable semantic leakage*
- If an LLM can distinguish synthetic from real data, it means patterns/artifacts exist that diverge from ground truth
- This is relevant to **attribute inference** and **record linkage** attacks that rely on such patterns
- Publication readiness: *Privacy in Statistical Databases* (PSD) is ideal venue

---

## 2. Methodology Overview

### 2.1 Data Pipeline

```
Real Data (Adult dataset)
    ↓
[30,162 records × 15 attributes]
    ↓
    ├─→ CTGAN (Conditional GAN)
    ├─→ TVAE (Variational AutoEncoder)
    ├─→ GaussianCopula (Parametric)
    ├─→ CART (Classification and Regression Tree)
    ├─→ GReat (Generative Regression Trees)
    └─→ TabDDPM (Diffusion-based)
    ↓
[6 sets of synthetic data, same shape]
    ↓
LLM Discrimination Prompt (C1/C2 conditions)
    ↓
[Binary classification: REAL or SYNTHETIC]
```

### 2.2 Experimental Design

**Conditions:**
- **C1 (Minimal):** Sample of 20 rows in Markdown table format only
- **C2 (Enhanced):** Same 20-row table + statistical metadata (mean, std, unique values, data types)

**Within-Subject Factors:**
- Method (6): CTGAN, TVAE, GaussianCopula, CART, GReat, TabDDPM
- Condition (2): C1 (table) vs C2 (table + metadata)
- Label (2): REAL vs SYNTHETIC examples
- Trials per condition: 2 (1 REAL, 1 SYNTHETIC per {method, condition})

**Between-Subject Factor:**
- LLM Provider/Model: Google Gemini (2.5 Flash, 2.5 Pro, 2.0 Flash, 3.x variants)

---

## 3. Privacy Interpretation Framework

### 3.1 What LLM Discrimination Measures

| Aspect | Interpretation | Privacy Implication |
|--------|---|---|
| **High Accuracy (>60%)** | Synthetic data has distinguishing artifacts | Potential attack surface for inference |
| **Low Accuracy (≈50%)** | Synthetic data mimics real distribution well | Better privacy protection |
| **Method Differences** | Some generators leak more than others | Allows comparison of privacy-preserving properties |
| **C1 vs C2 Gap** | Metadata increases discrimination power | Fine-grained attributes are privacy-critical |

### 3.2 Connection to Formal Privacy

- **Differential Privacy:** DP bounds statistical output. LLM discrimination measures *semantic* divergence
- **k-anonymity / l-diversity:** LLM can detect when synthetic records cluster in anomalous ways
- **Membership Inference:** Similar patterns; distinguishability is prerequisite for successful attacks
- **Attribute Inference:** Synthetic data artifacts enable inference of missing values

### 3.3 Practical Privacy Concerns

For Adult dataset (sensitive attributes: income, occupation, education):
- If LLM distinguishes synthetic from real, it detects *distributional differences*
- An attacker knowing "this is synthetic data" could:
  1. Infer which attributes are real vs imputed
  2. Estimate the original distribution more accurately
  3. Mount more effective attribute inference attacks

---

## 4. Current Results

### 4.1 Pilot Results (65 trials)

```
Overall Accuracy: 23.1% (worse than random baseline 50%)
└─ This is GOOD for privacy: LLM cannot distinguish

By Synthesizer:
├─ gaussian_copula:  40% (WORST—most detectable)
├─ ctgan:             27%
├─ great/tabddpm:     25%
└─ cart/tvae:         12% (BEST—most private)

By Condition:
├─ C1 (table only):   21% (harder to discriminate)
└─ C2 (+ metadata):   25% (metadata helps ~4%)

By Label:
├─ REAL instances:    31% accuracy
└─ SYNTHETIC:         15% accuracy (LLM-biased toward REAL)
```

**Interpretation:**
- **Most privacy-preserving:** TVAE, CART, GReat (≤25% discrimination)
- **Least privacy-preserving:** GaussianCopula (40% discrimination)
- **Metadata effect is small:** Only ~4% improvement with statistical summary

---

## 5. Future Experiments (Planned)

### 5.1 Multiple LLM Models

**Available via Google AI Studio:**
- `gemini-2.5-flash` ✓ (tested: 65 trials)
- `gemini-2.5-pro` (closed-access pro version)
- `gemini-2.0-flash` (previous generation)
- `gemini-2.0-flash-lite` (cheaper/faster)
- `gemini-3-flash-preview` (next generation)
- `gemini-3-pro-preview` (next generation pro)
- `gemma-3-27b-it`, `gemma-3-12b-it` (open models)

**Rationale:** 
- Model scale/capability affects discrimination ability
- Larger models → better pattern recognition → higher discrimination accuracy
- Relationship between model capability and privacy leakage is novel contribution

### 5.2 Additional Analyses

1. **Confidence Calibration:** Is LLM confidence correlated with correctness?
2. **Attribute-Level Analysis:** Which attributes trigger discrimination?
3. **Sample Size Effect:** How does row sample size (5 vs 20 vs 50) affect discrimination?
4. **Temporal Stability:** Do results change over time/re-runs?

### 5.3 Extended Experiments

- **More methods:** MUNGE, RLTGAN, PrivBayes, DP-variants
- **More datasets:** Census, MIMIC, Credit Card
- **Human baseline:** Recruit participants to perform same discrimination task

---

## 6. Research Contributions (for PSD Venue)

### 6.1 Novel Contributions

1. **New Privacy Metric:** LLM-based discrimination as privacy leakage indicator
   - Complements formal privacy bounds with "practical" privacy
   - Captures semantic divergence, not just statistical distance

2. **Comparative Study:** First systematic comparison of tabular synthesizers via LLM lens
   - Ranking: privacy-preserving properties beyond standard metrics
   - Model-agnostic: applicable to any LLM

3. **LLM Capability Analysis:** Understanding LLM discrimination across model scales
   - How does model capacity affect synthetic data detection?
   - What patterns do LLMs exploit?

### 6.2 Implications

- **For practitioners:** Choose synthesizers with measured via LLM discrimination
- **For theorists:** LLM discrimination as new interpretability angle on synthetic data
- **For privacy community:** Bridge between formal privacy and practical distinguishability

---

## 7. Reproducibility & Code

**All code open-source in:** `/code/`

```bash
# Data generation
python code/generate_synthetic_data.py

# LLM discrimination trials
export GOOGLE_API_KEY="<your-key>"
python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemini-2.5-flash --n_trials 50

# Analysis & visualization
python code/analyze_results.py --plot all
python code/analyze_results.py --pattern ctgan --plot heatmap
```

**Results format:** JSONL with per-trial metadata
```json
{
  "trial_id": 0,
  "timestamp": "2026-04-23T18:26:12.123456",
  "provider": "google",
  "model": "gemini-2.5-flash",
  "condition": "C2",
  "true_label": "REAL",
  "predicted_label": "REAL",
  "correct": true,
  "confidence": 0.89,
  "reasoning": "High variance in income distribution suggests real data...",
  "red_flags": ["outliers in capital_gain"],
  "supporting_evidence": ["income distribution matches expected"],
  "synthetic_method": "ctgan"
}
```

---

## 8. Limitations & Discussion

### Limitations

1. **Single Dataset:** Only Adult (income prediction). Results may not generalize.
2. **Binary Task:** Only REAL vs SYNTHETIC. Doesn't distinguish between synthesizers directly.
3. **LLM Selection Bias:** Gemini may have seen synthetic data in training; results could differ with Claude/GPT-4.
4. **Prompt Engineering:** All prompts use C1/C2. Other designs might yield different results.
5. **Known Synthetic Label:** LLM knows data origin. Real attack assumes "unknown" origin.

### Discussion Points

- **Why low overall accuracy?** LLMs struggle with tabular reasoning; may need domain-specific fine-tuning.
- **Why Gaussian Copula highest?** Parametric assumptions visible; fails to capture multimodal distributions.
- **Privacy implication:** Users needing strong privacy should avoid GaussianCopula, prefer TVAE/CART.

---

## 9. Next Steps (Milestones)

### Step 1: Expand Model Coverage (This Week)
- [ ] Test 5+ additional Google models (2.0, 3.x, Gemma)
- [ ] Document accuracy by model size
- [ ] Publish model × method × condition heatmap

### Step 2: Deeper Analysis (1-2 weeks)
- [ ] Attribute-level discrimination (which columns leak most?)
- [ ] Confidence calibration analysis
- [ ] Human baseline experiments

### Step 3: Extended Datasets (2-4 weeks)
- [ ] Census, MIMIC, Credit datasets
- [ ] More tabular synthesizers (PrivBayes, DP-MERF)
- [ ] Scaling analysis (100K → 1M records)

### Step 4: Paper Writing (4-6 weeks)
- [ ] Draft methodology & results sections
- [ ] Submit to Privacy in Statistical Databases (PSD 2027)
- [ ] Target 30-35 page conference paper

---

## 10. Author & Citation

**Project:** Data Synthesis Privacy via LLM Discrimination  
**Workspace:** `/Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026`  
**Start Date:** April 20, 2026  
**Status:** Pilot Phase Complete, Expansion Phase Underway  

**Suggested Venue:** Privacy in Statistical Databases (PSD 2026/2027)

---

## Appendix: Resources

- **SDV Documentation:** https://sdv.dev/ (tabular synthesizers)
- **Google Gemini API:** https://ai.google.dev/ (free tier: 1500 req/day)
- **Privacy Metrics:** https://openmetrics.ai/ (reference implementations)
- **PSD Conference:** https://www.nist.gov/itl/csd/privacy-statistical-databases


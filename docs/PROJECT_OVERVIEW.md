# LLM-as-Discriminator for Synthetic Data Privacy
## Project Overview & Roadmap

**Status:** Pilot Phase Complete ✓ | Expansion Phase Ready 🚀  
**Last Updated:** April 23, 2026  
**Target Venue:** Privacy in Statistical Databases (PSD 2026/2027)

---

## 1. Quick Summary

### What We're Doing
We're testing whether **Large Language Models can distinguish real from synthetic tabular data**. The ability of an LLM to discriminate is a new lens for measuring privacy leakage in synthetic data generators.

### Key Innovation
- **Traditional metrics:** DP-epsilon, k-anonymity, membership inference
- **Our contribution:** LLM-based discrimination as privacy metric
- **Why it matters:** Captures semantic divergence, detects human-readable patterns that might enable attacks

### Current Status
✅ **Pilot: 65 trials** across 6 synthesizers with 1 LLM (Gemini 2.5 Flash)
🚀 **Phase 2: Ready to test 5+ additional models** for comparative analysis
📝 **Paper: Outline drafted**, ready for expansion with multi-model results

---

## 2. Findings So Far (Pilot Phase)

### Overall Accuracy: 23.1% (Good for Privacy!)
```
Ideal result: 50% accuracy (can't distinguish = perfect privacy)
Our result: 23.1% accuracy (LLM worse than random guessing)
Worst method: 40% (GaussianCopula - most detectable)
Best method: 12% (TVAE - most private)
```

### Privacy Rankings
| Rank | Method | Accuracy | Privacy Rating |
|------|--------|----------|---|
| 🥇 | TVAE | 12% | EXCELLENT |
| 🥇 | CART | 12% | EXCELLENT |
| 🥈 | GReat | 25% | GOOD |
| 🥈 | TabDDPM | 25% | GOOD |
| 🥉 | CTGAN | 27% | GOOD |
| ❌ | GaussianCopula | 40% | POOR |

### Metadata Effect
- **C1 (table only):** 21% accuracy
- **C2 (table + stats):** 25% accuracy
- **Impact:** Only 4% improvement (metadata helps, but not dramatically)

---

## 3. Project Timeline

### ✅ COMPLETED (Apr 20-23)
- Data pipeline: Download Adult dataset (30,162 records)
- Synthetic data generation: 6 methods implemented
- LLM discrimination: End-to-end framework built
- Pilot results: 65 trials, 6 synthesizers, 1 LLM
- Visualization: 5 publication-quality plots
- Documentation: Analysis guides, methodology diagrams

**Code Files Created:**
```
code/
├── generate_synthetic_data.py   (Synthetic data generation)
├── evaluate.py                  (LLM discrimination trials)
├── prompt_builder.py            (Prompt formatting C1/C2)
├── analyze_results.py           (Analysis & visualization)
├── metadata.py                  (Data introspection)
└── app.py                       (Streamlit interface - optional)

docs/
├── PRIVACY_RESEARCH_SUMMARY.md           (This research for PSD)
├── METHODOLOGY_DIAGRAM.md                (System architecture)
├── MULTI_MODEL_EXPERIMENT_PLAN.md        (Phase 2 plan)
├── RESULTS_ANALYSIS_GUIDE.md             (How to interpret)
├── ANALYSIS_QUICK_START.md               (Quick reference)
└── FREE_API_SETUP.md                     (API setup)

data/
├── real/adult.csv                        (30,162 × 15)
└── synthetic/
    ├── ctgan/synthetic.csv
    ├── tvae/synthetic.csv
    ├── gaussian_copula/synthetic.csv
    ├── cart/synthetic.csv
    ├── great/synthetic.csv
    └── tabddpm/synthetic.csv

results/
├── llm_results_*.jsonl          (Per-trial results)
├── llm_results_*_summary.csv    (Aggregated stats)
└── plots/
    ├── 01_accuracy_by_method.png
    ├── 02_heatmap_method_x_condition.png
    ├── 03_confidence_distribution.png
    ├── 04_grouped_accuracy.png
    └── 05_confidence_vs_correctness.png
```

### 🚀 PHASE 2: MULTI-MODEL EXPANSION (This Week)

**Objective:** Test 5+ additional Google models to understand model capability's relationship to discriminability

```
Timeline: ~3-5 days
API Budget: ~4,000 calls (within free tier 1500/day)
Target Models:
  ├─ gemini-2.5-pro (pro version, high priority)
  ├─ gemini-2.0-flash (generational baseline)
  ├─ gemini-3-flash-preview (new generation fast)
  ├─ gemini-3-pro-preview (new generation large)
  └─ gemma-3-27b-it (open model variant)

Commands:
  python code/evaluate.py --real data/real/adult.csv \\
    --provider google \\
    --compare-models gemini-2.5-pro gemini-2.0-flash ... \\
    --n_trials 50
```

**Expected Outputs:**
- Model × Method × Condition heatmap (3D analysis)
- Model capability vs discrimination accuracy plot
- Ranking of models by capability and privacy measures

### 📊 PHASE 3: EXTENDED ANALYSIS (1-2 weeks after Phase 2)

**Objective:** Deep dive into patterns, scaling laws, and human baseline

```
Analysis Tasks:
├─ Scaling curve: Parameter count vs accuracy
├─ Attribute analysis: Which columns leak most?
├─ Confidence calibration: Is LLM certain when correct?
├─ Human baseline: Can people discriminate better?
├─ Few-shot learning: Train LLM with examples, remeasure
└─ Attack simulation: Use discrimination for membership inference

Code:
  python code/analyze_results.py --3d-analysis
  python code/model_comparison.py --scaling-curve
  python code/human_baseline.py --interface streamlit
```

### ✍️ PHASE 4: PAPER WRITING (2-4 weeks after Phase 3)

**Objective:** Draft publication-ready paper for Privacy in Statistical Databases

```
Section Structure:
├─ 1. Introduction (why privacy in synthetic data matters)
├─ 2. Related Work (DP, k-anon, membership inference, synthetic data)
├─ 3. Methodology (LLM discrimination as privacy metric)
├─ 4. Experiments (6 synthesizers × 5+ models × detailed analysis)
├─ 5. Results (rankings, scaling curves, implications)
├─ 6. Discussion (why certain methods leak, practical advice)
└─ 7. Conclusion & Future Work

Timeline:
  Week 1: Draft methods & experiments sections
  Week 2: Complete results & draft figures
  Week 3: Write discussion & conclusions
  Week 4: Revise, polish, submit to PSD

Target Length: 30-35 pages (conference format)
```

---

## 4. How to Run Next Steps

### Option A: Test One New Model (30 minutes)
```bash
export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026

# Test gemini-2.5-pro
python code/evaluate.py --real data/real/adult.csv \
  --provider google \
  --model gemini-2.5-pro \
  --n_trials 50

# Analyze results
python code/analyze_results.py
```

### Option B: Test All Tier-1 Models (3-5 hours)
```bash
export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026

# Run all models in sequence (with automatic pauses)
python code/evaluate.py --real data/real/adult.csv \
  --provider google \
  --compare-models gemini-2.5-pro gemini-2.0-flash \
                   gemini-3-flash-preview gemini-3-pro-preview \
                   gemma-3-27b-it \
  --n_trials 50

# After completion, analyze all results together
python code/analyze_results.py --plot all
python code/model_comparison.py --report-type summary
```

### Option C: Run Specific Synthetic Methods Only
```bash
# Test only TVAE and CTGAN (fastest methods)
python code/evaluate.py --real data/real/adult.csv \
  --provider google \
  --methods tvae ctgan \
  --model gemini-2.5-pro \
  --n_trials 30
```

---

## 5. Key Files & Quick Reference

### For Running Experiments
| File | Purpose | Command |
|------|---------|---------|
| `code/evaluate.py` | Main experiment runner | `python code/evaluate.py --real ... --provider google --compare-models ...` |
| `code/generate_synthetic_data.py` | Generate synthetic data | `python code/generate_synthetic_data.py` |

### For Analysis
| File | Purpose | Command |
|------|---------|---------|
| `code/analyze_results.py` | Analyze & visualize | `python code/analyze_results.py --plot all` |
| `code/model_comparison.py` | Model comparative analysis | `python code/model_comparison.py --scaling-curve` |

### For Documentation
| File | Purpose |
|------|---------|
| `PRIVACY_RESEARCH_SUMMARY.md` | Research context for PSD venue |
| `METHODOLOGY_DIAGRAM.md` | System architecture diagrams |
| `MULTI_MODEL_EXPERIMENT_PLAN.md` | Detailed Phase 2 plan |
| `RESULTS_ANALYSIS_GUIDE.md` | How to interpret results |
| `ANALYSIS_QUICK_START.md` | 2-minute quick reference |

### For Data
| Path | Contents |
|------|----------|
| `data/real/adult.csv` | Original Adult dataset (30,162 × 15) |
| `data/synthetic/{method}/synthetic.csv` | Synthetic data per method |
| `results/llm_results_*.jsonl` | Per-trial LLM results (JSON lines) |
| `results/plots/` | PNG visualizations |

---

## 6. Key Insights & Hypotheses

### Insight 1: Not All Synthetic Methods Are Equal
Different generators have different privacy profiles:
- **TVAE/CART** preserve privacy well (12% detectability)
- **GaussianCopula** leaks privacy (40% detectability)
- **CTGAN/GReat/TabDDPM** middleground (~25%)

**Implication:** Users should choose synthesizers based on privacy requirements, not just other metrics.

### Insight 2: Metadata Helps Little (4%)
Adding statistical summaries (C2 vs C1) improves accuracy only marginally.

**Implication:** Fine-grained attributes are harder to protect than summary statistics. Table-level privacy is achievable; attribute-level is harder.

### Hypothesis 1: Model Size Matters
Larger/newer models should discriminate better (higher accuracy).
- Gemini 2.0 < 2.5 < 3.0
- Pro > Flash > Flash-Lite

### Hypothesis 2: Scaling Laws Apply
Privacy leakage might follow a predictable curve with model parameters.

```
Expected:
  1B params → ~15% accuracy
  10B params → ~25% accuracy
  100B+ params → ~35-40% accuracy
```

### Hypothesis 3: Method Ranking Stable
The relative ordering (TVAE < CTGAN < GC) should persist across models.

---

## 7. Research Contribution to PSD

### Novel Contributions

1. **New Privacy Metric**
   - First systematic study of LLM-as-discriminator for synthetic data
   - Complements formal privacy (DP, k-anon) with practical detection
   - Model-agnostic evaluation framework

2. **Comparative Synthetic Data Study**
   - Ranked 6 tabular synthesizers by privacy
   - Analyzed across multiple dimensions (method, condition, LLM capability)
   - Provides practitioner guidance for synthesizer selection

3. **Model Capability Analysis**
   - Understanding how LLM size/generation affects synthetic data detection
   - Scaling laws for privacy leakage
   - Practical threshold identification ("acceptable" privacy level)

### Relevance to PSD
- **Privacy Focus:** Core to PSD's mission
- **Tabular Data:** PSD audience works with Census, health, financial data
- **Practical Impact:** Direct guidance for practitioners using synthetic data for statistical disclosure control
- **Novel Methodology:** LLM-based evaluation is fresh contribution to privacy literature

---

## 8. Success Metrics

### Phase 2 Success
- [ ] 5+ models tested with clean results
- [ ] Model ranking established
- [ ] Generational improvement quantified
- [ ] Scaling curve plotted

### Phase 3 Success
- [ ] Attribute-level analysis complete
- [ ] Human baseline established
- [ ] Confidence calibration assessed
- [ ] Attack simulations run

### Phase 4 Success
- [ ] Draft paper 15+ pages
- [ ] All figures created
- [ ] Full reproducibility package ready
- [ ] Submitted to PSD 2027

---

## 9. Contact & Questions

**Project Lead:** [Your name]  
**Workspace:** `/Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026`  
**Git Repo:** (Link if version-controlled)  
**Status Updates:** Check MULTI_MODEL_EXPERIMENT_PLAN.md for weekly progress

---

## 10. Appendix: Available Commands

### Quick Commands for Common Tasks

```bash
# Set API key and activate venv (always do this first!)
export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
source .venv/bin/activate

# Test single model
python code/evaluate.py --real data/real/adult.csv --provider google --model gemini-2.5-pro --n_trials 50

# Test multiple models
python code/evaluate.py --real data/real/adult.csv --provider google \
  --compare-models gemini-2.5-pro gemini-2.0-flash gemini-3-flash-preview

# Generate quick summary
python code/analyze_results.py

# Generate all plots
python code/analyze_results.py --plot all

# Filter by method
python code/analyze_results.py --pattern ctgan --plot heatmap

# Model comparison
python code/model_comparison.py --report-type summary

# Check API usage (Linux/Mac)
find results/ -name "*.jsonl" -newermt "today" | wc -l

# Resume interrupted run
python code/evaluate.py --real data/real/adult.csv --provider google \
  --model gemini-2.5-pro --resume-from 35 --n_trials 50
```

---

## 11. Next Immediate Actions

### TODO for Today/Tomorrow
- [ ] Read PRIVACY_RESEARCH_SUMMARY.md (understand venue/context)
- [ ] Read MULTI_MODEL_EXPERIMENT_PLAN.md (detailed Phase 2 plan)
- [ ] Run: `python code/evaluate.py --real data/real/adult.csv --provider google --model gemini-2.5-pro --n_trials 50`
- [ ] Run: `python code/analyze_results.py --plot all`
- [ ] Review new plots in `results/plots/`

### TODO for This Week
- [ ] Complete Phase 2a (test 5 Tier-1 models)
- [ ] Generate comparative analysis report
- [ ] Draft paper Section 3 (methodology) & Section 4 (experiments)

### TODO for This Month
- [ ] Complete Phase 2b (scaling analysis)
- [ ] Complete Phase 3 (extended analysis)
- [ ] Draft complete paper
- [ ] Prepare for PSD submission

---

**Last Status:** April 23, 2026 | Pilot complete, Phase 2 ready to launch 🚀


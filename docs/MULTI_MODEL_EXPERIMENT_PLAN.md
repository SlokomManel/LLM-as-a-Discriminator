# Multi-Model Experiment Plan
## Testing LLM-as-Discriminator Across Different Google Models

---

## 1. Available Models & Selection

### Recommended Models for Testing (Priority Order)

```
TIER 1 - HIGH PRIORITY (Test immediately)
├─ gemini-2.5-pro
│  └─ More capable than 2.5-flash
│  └─ Should show higher discrimination accuracy
│  └─ Costs same in free tier
│
├─ gemini-2.0-flash
│  └─ Previous generation
│  └─ Baseline for generational comparison
│
└─ gemini-3-flash-preview
   └─ Newest model
   └─ Understanding cutting-edge capability gap

TIER 2 - MEDIUM PRIORITY (If time permits)
├─ gemini-2.0-flash-lite
│  └─ Smaller model, faster inference
│  └─ Trade-off between speed and accuracy
│
├─ gemini-3-pro-preview
│  └─ Large cutting-edge model
│  └─ Expected to show highest discrimination
│
└─ gemma-3-27b-it
   └─ Open model from Google
   └─ Model size: 27B parameters
   └─ Different training/alignment

TIER 3 - OPTIONAL (Lower priority, if budget allows)
├─ gemma-3-12b-it, gemma-3-4b-it, gemma-3-1b-it
│  └─ Smaller variants
│  └─ Understand scaling laws of discrimination ability
│
└─ Other specialized models (robotics, vision, etc.)
   └─ Not recommended for tabular discrimination
```

---

## 2. Experimental Plan

### Phase 2a: Core Model Comparison (This Week)
**Target:** Test 5 models systematically

| Model | Generation | Capability | Priority | Rationale |
|-------|-----------|-----------|----------|-----------|
| gemini-2.5-flash | 2.5 | Best free | DONE ✓ | Baseline established |
| gemini-2.5-pro | 2.5 | Better | HIGH | More capable version |
| gemini-2.0-flash | 2.0 | Older | HIGH | Generational comparison |
| gemini-3-flash-preview | 3.0 | Newest | HIGH | Cutting edge |
| gemini-3-pro-preview | 3.0 | Largest | MEDIUM | Max capability |

**Commands to Execute:**
```bash
export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
source .venv/bin/activate

# Model 1: gemini-2.5-pro (Already tested as fallback, now systematic)
python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemini-2.5-pro --n_trials 50 \
  --suffix "phase2a_25pro"

# Model 2: gemini-2.0-flash (Previous generation)
python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemini-2.0-flash --n_trials 50 \
  --suffix "phase2a_20flash"

# Model 3: gemini-3-flash-preview (Next generation, fast)
python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemini-3-flash-preview --n_trials 50 \
  --suffix "phase2a_3flash"

# Model 4: gemini-3-pro-preview (Next generation, large)
python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemini-3-pro-preview --n_trials 50 \
  --suffix "phase2a_3pro"

# Model 5: gemma-3-27b-it (Open model, large)
python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemma-3-27b-it --n_trials 50 \
  --suffix "phase2a_gemma27b"
```

**Expected Outcomes:**
- Models ranked by discrimination accuracy
- Generational improvements visible (2.0 → 2.5 → 3.0)
- Size/capability correlation analyzed
- "Diminishing returns" curve (does bigger = better?)

**Timeline:** ~2-3 hours per model (including wait for API)

---

### Phase 2b: Scaling Analysis (Following week)
**Target:** Understand model scaling laws

Test smaller models to see scaling curve:
```bash
# Small models (Gemma variants)
python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemma-3-12b-it --n_trials 30 \
  --suffix "scaling_12b"

python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemma-3-4b-it --n_trials 30 \
  --suffix "scaling_4b"

python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemma-3-1b-it --n_trials 30 \
  --suffix "scaling_1b"
```

**Analysis:** Plot discrimination accuracy vs model size (parameters)

---

## 3. Analysis Strategy

### After Each Model Run

```bash
# Generate fresh analysis including new model
python code/analyze_results.py \
  --pattern "phase2a" \
  --plot all \
  --output results/plots_phase2a/

# Generate model comparison plot
python code/analyze_results.py \
  --compare_models gemini-2.5-flash gemini-2.5-pro gemini-2.0-flash \
  --plot comparison_bar
```

### Final Comparative Analysis

```bash
# Create master table: accuracy by [Method × Condition × Model]
python code/analyze_results.py \
  --compare_all_models \
  --output results/comparative_analysis/ \
  --format csv markdown
```

**Output:** 
- 3D heatmap: Method × Condition × Model
- Line plot: Accuracy vs Model capability
- Table: Rankings across all dimensions

---

## 4. Code Changes Needed

### 4.1 Update evaluate.py to handle multiple models in batch

Add flag for easy batch testing:

```python
# NEW: --compare-models flag
python code/evaluate.py --real data/real/adult.csv \
  --compare-models gemini-2.5-flash gemini-2.5-pro gemini-2.0-flash \
  --n_trials 20

# This runs: Method × Condition × Label × 3_models = 24 × 3 = 72 trials per method
```

### 4.2 Update analyze_results.py for model comparison

```python
# NEW: Compare accuracy across models
python code/analyze_results.py --compare-models --plot model_accuracy

# NEW: 3D analysis
python code/analyze_results.py --3d-heatmap method_condition_model
```

### 4.3 New utility: model_comparison.py

Helper script specifically for managing multi-model experiments:

```python
# NEW FILE: code/model_comparison.py

def compare_model_accuracy(models: list[str], pattern: str = None):
    """
    Load results for multiple models and create comparison DataFrame
    
    Output:
        DataFrame with columns:
        - synthetic_method
        - condition  
        - model
        - accuracy
        - trials_count
    """
    ...

def plot_model_ranking(df):
    """Create bar plot: Accuracy by Model, colored by method"""
    ...

def plot_scaling_curve(df):
    """Create line plot: Accuracy vs Model Capability"""
    ...
```

---

## 5. Expected Results & Hypotheses

### Hypothesis 1: Larger Models Discriminate Better
**Prediction:** Pro > Flash > Flash-Lite (within generation)
```
Gemini 2.5 Pro: 35-45% accuracy (vs Flash: 23%)
Gemini 3.0 Pro: 40-50% accuracy (vs 3.0 Flash: 30-35%)
```

**Reasoning:** 
- Larger models have more parameters for pattern detection
- Better instruction-following → more careful discrimination
- But: Still bounded by real synthetic similarity

### Hypothesis 2: Generational Improvements
**Prediction:** 2.0 < 2.5 < 3.0
```
Gemini 2.0 Flash: 15-20% accuracy
Gemini 2.5 Flash: 23% accuracy ✓ (baseline)
Gemini 3.0 Flash: 25-30% accuracy
```

**Reasoning:**
- Newer models trained on more token diversity
- Better at detecting subtle distributional shifts
- Improved reasoning capabilities

### Hypothesis 3: Scaling Laws
**Prediction:** Log-linear relationship
```
1B parameters:  ~15% accuracy
4B parameters:  ~18% accuracy
12B parameters: ~22% accuracy
27B parameters: ~25% accuracy
Large models:   ~30-40% accuracy
```

**Reasoning:**
- Tabular reasoning is easier than NLP text tasks
- Even small models can detect synthetic artifacts
- Diminishing returns above certain size

### Hypothesis 4: Method Stability Across Models
**Prediction:** Ranking (TVAE < CART < CTGAN < GC) consistent across models
```
Tvae discriminability: 10-15% (all models)
Gaussian Copula:       35-50% (all models)
```

**Reasoning:**
- Fundamental differences in data quality persist
- All models should notice GaussianCopula artifacts
- TVAE semantic capture is model-independent

---

## 6. Monitoring & Checkpoints

### Daily Checkpoint

```bash
# Check current results
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026

# Count trials by model
echo "TRIALS BY MODEL:"
grep -h '"model"' results/llm_results_*.jsonl | sort | uniq -c

# Current accuracy snapshot
python code/analyze_results.py | grep "Overall Accuracy"

# Model-specific accuracy
python code/analyze_results.py --group-by model
```

### Weekly Summary

```bash
# Generate comparison report
python code/model_comparison.py \
  --report-type weekly \
  --output reports/week_$(date +%Y%m%d).md

# Update paper draft with latest results
python code/model_comparison.py \
  --report-type paper \
  --output paper/results_interim.txt
```

---

## 7. Budget & Rate Limits

### Google Free Tier (Active)
- **Limit:** 1,500 requests/day
- **Rate:** Unlimited (daily aggregate)
- **Cost:** FREE

### Capacity Analysis
```
Experiment Plan:
├─ Phase 2a (5 models × 50 trials each)
│  └─ 5 × 50 × 12 API calls per trial = 3,000 calls
│  └─ Spread over 3-4 days (750-1000/day) ✓ FITS in quota
│
├─ Phase 2b (3 models × 30 trials each)
│  └─ 3 × 30 × 12 = 1,080 calls  
│  └─ Spread over 2 days (540/day) ✓ FITS in quota
│
└─ Total: ~4,000 API calls ≈ 5-6 days of experimentation
```

### Management Tips
```bash
# Monitor API usage in real-time
while true; do
  echo "$(date): Files created today:"
  find results/ -name "*.jsonl" -newermt "today" | wc -l
  echo "Estimate: Each file ≈ 0-12 API calls"
  sleep 300  # Check every 5 minutes
done

# Graceful degradation if quota exceeded
# → Code automatically waits until next day
# → Results accumulated so far aren't lost
# → Resume with --resume-from flag
python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemini-2.5-pro \
  --resume-from 35 --n_trials 50
```

---

## 8. Documentation & Publication

### Interim Report (After Phase 2a)
File: `INTERIM_REPORT_PHASE2A.md`

Content:
- Results summary: 5 models × 6 methods × 2 conditions = 240 trials
- Accuracy rankings by model
- Generational comparison analysis
- Preliminary conclusions

### Final Report (After Phase 2b)
File: `FINAL_RESULTS_REPORT.md`

Content:
- Complete experimental results (all models)
- Model scaling analysis
- Privacy implications
- Recommendations for practitioners
- Ready for PSD submission

### Paper Sections to Draft
```
Section 3.1 (Experiments)
└─ Multiple baseline models tested
   Method comparison across Gemini versions
   
Section 4 (Results)
├─ Model rankings table
├─ Generational improvement plot
└─ Scaling analysis graph

Section 5 (Discussion)
├─ Why larger models correlate with higher discrimination
├─ Whether discrimination ability plateaus
└─ Practical implications: What threshold is "acceptable privacy"?
```

---

## 9. Quick Start Commands

### TODAY - Test one new model
```bash
export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026

# Test gemini-2.5-pro (pro version of what we used before)
python code/evaluate.py --real data/real/adult.csv \
  --provider google --model gemini-2.5-pro --n_trials 50
```

### THIS WEEK - Full Phase 2a
```bash
# Run all 5 tier-1 models (takes ~6 hours total)
bash scripts/run_phase2a.sh  # See section 10 below
```

### THIS MONTH - Paper Ready
After Phase 2b completes, have full comparative analysis for submission

---

## 10. Batch Script (scripts/run_phase2a.sh)

```bash
#!/bin/bash
# Run all Phase 2a models systematically

export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026

MODELS=(
  "gemini-2.5-pro"
  "gemini-2.0-flash"
  "gemini-3-flash-preview"
  "gemini-3-pro-preview"
  "gemma-3-27b-it"
)

for MODEL in "${MODELS[@]}"; do
  echo "════════════════════════════════════════════"
  echo "Testing: $MODEL"
  echo "════════════════════════════════════════════"
  
  .venv/bin/python code/evaluate.py \
    --real data/real/adult.csv \
    --provider google \
    --model "$MODEL" \
    --n_trials 50 \
    2>&1 | tee "logs/phase2a_${MODEL}.log"
  
  # Pause between models to avoid rate limit
  echo "Waiting 30 seconds before next model..."
  sleep 30
  
  # Analyze intermediate results
  echo "Current status:"
  .venv/bin/python code/analyze_results.py --pattern "phase2a" 2>/dev/null || true
done

echo "PHASE 2A COMPLETE!"
python code/model_comparison.py --report-type summary
```


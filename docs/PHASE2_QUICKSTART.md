# Phase 2: Multi-Model Testing
## Quick Start Guide (Copy & Paste Ready)

**Duration:** ~3-5 hours | **API Cost:** FREE | **Result Size:** ~400 trials

---

## Step 1: Verify Setup (5 minutes)

```bash
# Navigate to project directory
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026

# Activate virtual environment
source .venv/bin/activate

# Verify API key is available
echo "API Key set: $([ -z "$GOOGLE_API_KEY" ] && echo 'NO ❌' || echo 'YES ✓')"

# If not set, run this:
export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"

# Verify Python environment
.venv/bin/python --version

# Quick test: List available models (you can also just run this directly)
python << 'PYEOF'
import google.generativeai as g
import warnings
warnings.filterwarnings('ignore')

g.configure(api_key='AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY')
models = [m.name.replace('models/', '') for m in g.list_models() 
          if 'generateContent' in m.supported_generation_methods]
print(f"✓ Found {len(models)} models")
print("Sample:", models[:3])
PYEOF
```

---

## Step 2: Pick Your Testing Strategy

### Option A: Quick Test (Recommended First)
**Time:** 45 minutes | **Models:** 1 | **Trials:** 50  
**Output:** Single model results for comparison with baseline

```bash
export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
source .venv/bin/activate

echo "Testing gemini-2.5-pro..."
python code/evaluate.py \
  --real data/real/adult.csv \
  --provider google \
  --model gemini-2.5-pro \
  --n_trials 50

echo "✓ Complete! Analyze with:"
echo "python code/analyze_results.py --plot all"
```

---

### Option B: Standard Phase 2a (Recommended)
**Time:** 3-4 hours | **Models:** 5 (Priority tier) | **Trials:** 50 each  
**Output:** Comprehensive model comparison

```bash
#!/bin/bash
# Copy this entire block, paste into terminal, press Enter

export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
source .venv/bin/activate

# List of models to test (Tier 1 - priority)
MODELS=(
  "gemini-2.5-pro"
  "gemini-2.0-flash"
  "gemini-3-flash-preview"
  "gemini-3-pro-preview"
  "gemma-3-27b-it"
)

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              PHASE 2A: TESTING 5 MODELS                       ║"
echo "║        (This will take 3-4 hours. You can interrupt anytime)   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

START_TIME=$(date)
echo "Started: $START_TIME"
echo ""

for MODEL in "${MODELS[@]}"; do
  echo "════════════════════════════════════════════════════════════════"
  echo "Testing: $MODEL"
  echo "════════════════════════════════════════════════════════════════"
  
  python code/evaluate.py \
    --real data/real/adult.csv \
    --provider google \
    --model "$MODEL" \
    --n_trials 50
  
  if [ $? -eq 0 ]; then
    echo "✓ $MODEL COMPLETE"
  else
    echo "✗ $MODEL FAILED (continuing...)"
  fi
  
  # Pause before next model (be nice to API)
  if [ "$MODEL" != "${MODELS[-1]}" ]; then
    echo ""
    echo "Pausing 20 seconds before next model..."
    sleep 20
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ ALL MODELS TESTED!"
echo "════════════════════════════════════════════════════════════════"
echo "Started: $START_TIME"
echo "Ended: $(date)"
echo ""
echo "Next: Analyze all results together"
echo ""
```

---

### Option C: Extended Phase 2 (Maximum Coverage)
**Time:** 6-8 hours | **Models:** 10 (Tier 1 + 2) | **Trials:** 30 each  
**Output:** Comprehensive scaling analysis + model ranking

```bash
#!/bin/bash

export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
source .venv/bin/activate

MODELS=(
  # TIER 1 (High priority)
  "gemini-2.5-pro"
  "gemini-2.0-flash"
  "gemini-3-flash-preview"
  "gemini-3-pro-preview"
  "gemma-3-27b-it"
  # TIER 2 (Medium priority)
  "gemini-2.0-flash-lite"
  "gemma-3-12b-it"
  "gemma-3-4b-it"
  "gemini-2.5-flash-lite"
  "gemma-3-1b-it"
)

for MODEL in "${MODELS[@]}"; do
  python code/evaluate.py \
    --real data/real/adult.csv \
    --provider google \
    --model "$MODEL" \
    --n_trials 30
  sleep 15
done
```

---

## Step 3: Monitor Progress

### While Running
```bash
# In a separate terminal, monitor files being created
while true; do
  clear
  echo "Monitoring... (Ctrl+C to stop)"
  echo ""
  echo "Total result files:"
  ls -1 results/llm_results_*.jsonl | wc -l
  echo ""
  echo "Latest file:"
  ls -lht results/llm_results_*.jsonl | head -1
  echo ""
  echo "Trials completed (estimate):"
  wc -l results/llm_results_*.jsonl | tail -1
  echo ""
  sleep 30
done
```

### Check Intermediate Results
```bash
# Quick accuracy snapshot (while running or after)
.venv/bin/python code/analyze_results.py

# Generate plots
.venv/bin/python code/analyze_results.py --plot all
```

---

## Step 4: Analyze & Visualize Results

```bash
# Generate quick summary (text)
python code/analyze_results.py

# Generate all 5 visualizations (PNG files)
python code/analyze_results.py --plot all

# Generate model comparison
python << 'PYEOF'
import pandas as pd
from pathlib import Path
import json

# Load all results
results = []
for jsonl_file in Path("results").glob("llm_results_*.jsonl"):
    with open(jsonl_file) as f:
        for line in f:
            results.append(json.loads(line))

df = pd.DataFrame(results)

# Model-wise accuracy
print("═" * 60)
print("ACCURACY BY MODEL")
print("═" * 60)
model_acc = df.groupby("model")["correct"].agg(["sum", "count", "mean"]).sort_values("mean", ascending=False)
print(model_acc)
print()

# Method × Model
print("═" * 60)
print("ACCURACY BY METHOD × MODEL")
print("═" * 60)
method_model = df.groupby(["synthetic_method", "model"])["correct"].mean().unstack().sort_index()
print(method_model.round(3))
PYEOF
```

---

## Step 5: Generate Report

```bash
# Create comprehensive report
cat > results/PHASE_2A_REPORT.md << 'REPORT'
# Phase 2A Results Summary

Generated: $(date)

## Models Tested
$(grep -h '"model"' results/llm_results_*.jsonl | cut -d'"' -f4 | sort -u)

## Accuracy by Model
[See output from Step 4 above]

## Key Findings
- Insert main findings here
- Compare to baseline (23.1% from Phase 1)
- Identify best/worst models

## Next Steps
- Phase 3: Attribute-level analysis
- Human baseline testing
- Attack simulation
REPORT

echo "✓ Report template created at: results/PHASE_2A_REPORT.md"
```

---

## Common Issues & Fixes

### Issue: API Rate Limited
```
Error: "Error 429: Rate limit exceeded"
Solution: Wait 5-10 minutes for daily limit to reset, then continue
          The code automatically pauses between models to avoid this
```

### Issue: Model Name Not Found
```
Error: "models/xyz-123 is not found"
Solution: Check METHODOLOGY_DIAGRAM.md Section "Available Models"
          or run the model list verification in Step 1
```

### Issue: Quota Exceeded (Different from Rate Limit)
```
Error: "You exceeded your current quota"
Solution: This is account-level limit, not daily
          Check your Google AI Studio dashboard
          Free tier: 1500 requests per day maximum
```

### Issue: Interrupted Mid-Run
```
Solution: All results saved incrementally (JSONL format)
          Just re-run the same command - it will append new trials
          Completed trials won't be re-run
```

---

## Expected Output Snapshots

### After One Model (45 min)
```
files/  
  llm_results_ctgan_google_20260423_200000.jsonl (2.5 KB)
  llm_results_ctgan_google_20260423_200000_summary.csv (1.2 KB)
  llm_results_tvae_google_20260423_200500.jsonl (2.4 KB)
  llm_results_tvae_google_20260423_200500_summary.csv (1.3 KB)
  ... (other 4 synthetic methods)
  
Results:
  Overall accuracy: 18-35% (varies by model)
  By method: Accuracy rankings
```

### After All 5 Models (3-4 hours)
```
100+ result files
250-300 total trials
Complete model ranking established
Ready for comparative analysis

Analysis Output:
  plots/01_accuracy_by_method.png          (bar chart)
  plots/02_heatmap_method_x_condition.png  (method × condition heatmap)
  plots/03_confidence_distribution.png     (boxplot)
  plots/04_grouped_accuracy.png            (grouped bars by model)
  plots/05_confidence_vs_correctness.png   (scatter plot)
  
Text Summary:
  Overall Accuracy by Model: [table]
  Best Model: [X% accuracy]
  Scaling Analysis: [preliminary curve]
```

---

## Real Example: Run Complete Phase 2a

**Just copy-paste this entire block:**

```bash
#!/bin/bash
set -e  # Exit on error

export GOOGLE_API_KEY="AIzaSyDvhDUsOkdnHHM_Hy5wKPJxPsXpvQar5GY"
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
source .venv/bin/activate

echo "Phase 2A: Multi-Model Testing"
echo "Started: $(date)"
echo ""

for MODEL in gemini-2.5-pro gemini-2.0-flash gemini-3-flash-preview gemini-3-pro-preview gemma-3-27b-it; do
  echo "Testing $MODEL..."
  python code/evaluate.py --real data/real/adult.csv --provider google --model "$MODEL" --n_trials 50
  [ "$MODEL" != "gemma-3-27b-it" ] && sleep 20
done

echo ""
echo "All models tested! Generating analysis..."
python code/analyze_results.py --plot all

echo "✓ Complete!"
echo "Check results/plots/ for visualizations"
```

---

## Time Estimate Breakdown

| Task | Time | Notes |
|------|------|-------|
| Setup & verification | 5 min | One-time |
| Single model test (50 trials × 6 methods) | 45 min | ~150 API calls |
| 5 models test | 3-4 hours | 750 API calls total |
| Analysis & visualization | 5 min | Automatic |
| Report writing | 10 min | Optional |
| **Total (5 models)** | **~4 hours** | Within free tier quota |

---

## Success Criteria

After Phase 2a, you should have:

- [ ] **5 models tested** with clean results
- [ ] **Overall accuracy ranking** (which model discriminates best?)
- [ ] **300+ trials completed** (from initial 65)
- [ ] **5 new PNG visualizations** updated with multi-model data
- [ ] **Model scaling curve** (model size/generation vs discrimination)
- [ ] **Method stability** (are rankings consistent across models?)

---

## Next: Phase 3 (After Phase 2a Complete)

Once Phase 2a results are complete:

```bash
# Attribute-level analysis
python code/analyze_results.py --attribute-level

# Confidence calibration
python code/analyze_results.py --confidence-analysis

# Start human baseline
python code/app.py --mode human_baseline
```

---

**Ready? Pick Option A, B, or C above and paste the code! 🚀**


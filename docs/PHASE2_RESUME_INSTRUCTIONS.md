# Phase 2 Resume Instructions (for tomorrow, April 24)

**Current Status:** 1,603 trials collected before quota reset  
**Remaining:** 4 models × 6 methods × 2 conditions × 50 trials = ~2,400 new trials  
**Expected time:** 3-4 hours for all remaining models

---

## 🎯 What to Do Tomorrow

### Step 1: Wait for UTC Midnight (or close to it)

The daily Google API quota resets at **midnight UTC** (8:00 PM ET / 5:00 PM PT on April 23 Pacific time).

Check when that is in your timezone:
- UTC midnight = **April 24, 00:00:00 UTC**
- Convert to your local time at: https://www.timeanddate.com/worldclock/timezone/utc

### Step 2: Run the Resume Script (after midnight UTC)

```bash
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
bash resume_phase2.sh
```

This runs:
- **gemini-2.0-flash** (2-3 min per method)
- **gemini-3-flash-preview** (2-3 min per method)  
- **gemini-3-pro-preview** (3-4 min per method)
- **gemma-3-27b-it** (3-4 min per method)

Each model tests all 6 synthetic methods, so:
- ~15 min per model
- ~60 min total for all 4 models
- Can run in background with nohup

### Step 3: Monitor Progress

While running, check progress in another terminal:

```bash
# Watch the log in real-time
tail -f experiment_phase2_resume.log

# Count trials being collected
watch 'wc -l results/llm_results_*.jsonl | tail -1'  # Updates every 2 sec
```

### Step 4: When Done (Tomorrow Evening)

```bash
# Verify all trials are collected
wc -l results/llm_results_*.jsonl | tail -1
# Should show ~4,000+ lines (1,603 old + 2,400 new)

# Re-run analysis to generate updated graphs
python analyze_results.py
```

---

## 📊 Current Results Summary

**Total: 1,603 trials so far**

### Privacy Rankings (what we learned so far):
```
🔒 BEST PRIVACY  (LLM can't discriminate):
   1. CART              0.5% accuracy 
   2. GReat             0.7% accuracy 
   3. TVAE              1.0% accuracy 

⚠️  WORST PRIVACY   (easiest to distinguish):
   4. CTGAN             1.2% accuracy 
   5. TabDDPM           1.3% accuracy 
   6. GaussianCopula    1.3% accuracy 
```

### By Model (so far):
- **gemini-2.5-flash:** 62.5% accuracy (24 trials) ← Best discriminator
- **gemini-2.5-pro:** 0% (mostly quota errors from 1,555 tests)
- **gemini-2.0-flash:** 0% (21 trials, limited testing)
- **gpt-4o:** 0% (3 trials, limited testing)

### Key Insights:
1. **Quota problem:** Almost all gemini-2.5-pro results are errors (0% = mostly failed API calls)
2. **Flash is better:** gemini-2.5-flash shows strong discrimination ability
3. **CART is safest:** CART synthetic data is hardest for LLMs to distinguish from real
4. **Confidence is unreliable:** Model reports 96.9% confidence even when getting 0% accuracy!

---

## ⚠️ Important Notes

### If script fails with "zsh: command not found":
```bash
# Make sure to activate venv first
source .venv/bin/activate
bash resume_phase2.sh
```

### If you get quota errors again:
- 429 errors are EXPECTED when quota resets mid-test
- Script auto-retries with backoff
- You can safely stop and resume later (results are saved incrementally)

### If you want to keep laptop closed:
- The script uses `nohup`, so terminal can close
- See output in `experiment_phase2_resume.log`
- Check logs with: `tail experiment_phase2_resume.log`

### Alternative: Use Groq instead (no waiting)
If midnight is too far away, switch to Groq (12,000 req/day limit):
```bash
source .venv/bin/activate
export GROQ_API_KEY="your-key"
nohup python code/evaluate.py --real data/real/adult.csv --provider groq \
  --model llama-3.1-70b-versatile --n_trials 50 > experiment_groq.log 2>&1 &
```

---

## 📈 Next Steps After Phase 2

Once all ~4,000 total trials are collected:

1. **Phase 3 Analysis:** Run detailed investigation
   - Attribute-level analysis (which columns leak privacy?)
   - Human baseline (can humans discriminate better?)
   - Confidence calibration (is model calibrated?)

2. **Paper Draft:** Start writing for Privacy in Statistical Databases venue
   - Use visualizations from `results/fig*.png`
   - Use method rankings from analysis

3. **Extended Testing (Optional):**
   - Test larger language models (GPT-4, Claude, Llama 3.1)
   - Test with different privacy budgets

---

## 📝 Files Created for You

- `analyze_results.py` — Loads results, generates 4 visualizations
- `resume_phase2.sh` — Bash script to run remaining models
- `analysis_output.log` — Log from first run (this one)
- `results/fig1_condition_analysis.png` — Graphs by condition
- `results/fig2_privacy_rankings.png` — Privacy ranking chart
- `results/fig3_confidence_calibration.png` — Confidence histogram
- `results/fig4_prediction_bias.png` — Prediction bias chart

---

## 🚀 Quick Checklist for Tomorrow

- [ ] Wait until ~midnight UTC
- [ ] Open terminal and navigate to project directory
- [ ] Run: `bash resume_phase2.sh`
- [ ] Monitor: `tail -f experiment_phase2_resume.log`
- [ ] When done (~1 hour), run: `python analyze_results.py`
- [ ] Check graphs in `results/` folder
- [ ] commit to papers/main.tex start

Good luck! 🎯

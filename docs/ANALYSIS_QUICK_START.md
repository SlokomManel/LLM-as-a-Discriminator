# 🚀 Quick Start: Analyze Your Results

## In 30 Seconds

```bash
# See all your results summarized
python code/analyze_results.py

# See beautiful plots
python code/analyze_results.py --plot all
```

That's it! Two commands.

## What You'll Get

### Command 1: Text Summary
```
📊 LLM DISCRIMINATION RESULTS SUMMARY

📈 Overall Statistics:
  Total trials: 65
  Overall accuracy: 23.1%
  Average confidence: 31%

📋 By Synthetic Method:
                  Correct  Total Accuracy %  Avg Confidence
  gaussian_copula         4     10        40%          38.3%
  ctgan                   4     15        27%          25.0%
  great                   2      8        25%          47.8%
  tabddpm                 2      8        25%          47.8%
  cart                    2     16        12%          23.4%
  tvae                    1      8        12%          11.6%
```

### Command 2: 5 Beautiful Plots

| Plot | Shows | Use When |
|------|-------|----------|
| **Bar Chart** | Which method is hardest to detect | Comparing method quality |
| **Heatmap** | C1 vs C2 performance | Checking if metadata helps |
| **Grouped Bars** | Side-by-side method comparison | Presenting to audience |
| **Boxplot** | Confidence distribution | Understanding LLM certainty |
| **Scatter** | Confidence vs correctness | Analyzing calibration |

All saved to `results/plots/`

---

## Understanding the Numbers

**What does "Accuracy %" mean?**

For **gaussian_copula** with **40% (4/10)**:
- Ran 10 trials total
- LLM got 4 correct
- That's 40% accuracy

**Why is accuracy so low (23%)?**
- Random guessing = 50%
- Your result (23%) = slightly worse than guessing
- **Means:** Modern synthetic data is VERY realistic!

---

## Common Analyses

**"Show me just the Google results"**
```bash
python code/analyze_results.py --pattern google
```

**"Show me just CTGAN"**
```bash
python code/analyze_results.py --pattern ctgan
```

**"I only want the heatmap"**
```bash
python code/analyze_results.py --plot heatmap
```

**"Save plots somewhere else"**
```bash
python code/analyze_results.py --plot all --output my_paper/figures/
```

---

## The Raw Data

If you want to dive deeper:

```bash
# Read the raw JSON results
cat results/llm_results_ctgan_google_*.jsonl | head -1 | python -m json.tool
```

Each trial contains:
- `verdict` — LLM's answer (REAL/SYNTHETIC)
- `confidence` — LLM's certainty (0-100%)
- `reasoning` — Detailed explanation
- `red_flags` — Suspicious patterns LLM found
- `supporting_evidence` — Evidence of authenticity

---

## Next Steps

1. **Run full-scale trials** → Increase `--n_trials` to 50+
2. **Try different LLM** → Switch from Google to OpenAI/Anthropic
3. **Add human baseline** → Compare against human discrimination
4. **Dive into reasoning** → Analyze why LLM fails at certain methods
5. **Write paper** → Include these plots!

---

See **RESULTS_ANALYSIS_GUIDE.md** for detailed explanations.

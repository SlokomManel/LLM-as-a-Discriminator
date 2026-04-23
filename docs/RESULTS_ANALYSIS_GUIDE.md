# Understanding LLM Discrimination Results

## 📊 How the Results Table is Calculated

### Example from Your Data:
```
                            sum  count  mean
synthetic_method condition                  
ctgan            C1           2      2   1.0
gmail_copula      C1           2      2   1.0
tvae             C1           0      2   0.0
```

### Breaking It Down:

**For CTGAN C1 (sum=2, count=2, mean=1.0):**
- **count**: Total number of trials = **2**
  - 1 trial with REAL data
  - 1 trial with SYNTHETIC data
- **sum**: Number of correct predictions = **2** 
  - LLM correctly identified both REAL and SYNTHETIC
- **mean**: Accuracy = sum ÷ count = 2 ÷ 2 = **1.0 (100%)**

**For TVAE C1 (sum=0, count=2, mean=0.0):**
- **count**: 2 trials total
- **sum**: 0 correct predictions
- **mean**: 0 ÷ 2 = **0.0 (0%)** — LLM got both wrong

### Key Metrics Explained:

| Metric | Meaning | Formula |
|--------|---------|---------|
| **sum** | # correct predictions | Count of True/True results |
| **count** | Total trials | REAL trials + SYNTHETIC trials |
| **mean** | Accuracy percentage | sum ÷ count |
| **confidence** | LLM's self-reported certainty | 0-100% |

---

## 📈 Visualization Types & What They Show

### 1️⃣ **Horizontal Bar Chart** (Accuracy by Method)
**Shows:** Which synthetic methods are easiest/hardest to detect
- **Green bars** = Highest accuracy (best discrimination)
- **Red bars** = Lowest accuracy (worst discrimination)
- **Label shows:** `12% (1/8)` = 1 correct out of 8 trials

**Key insight:** 
- `gaussian_copula` at 40% is best (most detectable as fake)
- `cart` & `tvae` at 12% are hardest (most realistic-looking)

### 2️⃣ **Heatmap** (Methods × Conditions)
**Shows:** How conditions (C1 vs C2) affect discrimination accuracy
- **Green (bright)** = High accuracy
- **Red (dark)** = Low accuracy
- **Rows** = Synthetic methods
- **Columns** = C1 (table only) vs C2 (table + metadata)

**Key insight:**
- **gaussian_copula C2**: 50% (best performance)
- **tvae C1**: 0% (worst performance)
- **C2 generally > C1** = Metadata helps LLM discriminate

### 3️⃣ **Grouped Bar Chart** (Methods × Conditions Comparison)
**Shows:** Side-by-side comparison within each method
- **Blue bars** = C1 performance
- **Orange bars** = C2 performance
- **Height** = Accuracy %

**Key insight:** 
- Where C2 (orange) is noticeably taller = metadata provides critical info
- gaussian_copula shows biggest C1→C2 improvement (33% → 50%)

### 4️⃣ **Confidence Distribution** (Boxplot)
**Shows:** Spread of LLM confidence levels by method
- **Box** = middle 50% of confidence values
- **Line inside box** = median confidence
- **Whiskers** = range (min-max confidence)

**Key insight:**
- Some methods LLM is confident in (high median)
- Others LLM is very uncertain (low median)

### 5️⃣ **Scatter Plot** (Confidence vs Correctness)
**Shows:** Does LLM confidence correlate with correctness?
- **Blue dots** = REAL data trials
- **Red dots** = SYNTHETIC data trials
- **Y-axis = 100** = Correct guess
- **Y-axis = 0** = Wrong guess
- **X-axis** = LLM's reported confidence (%)

**Key insight:**
- Points at top = correct predictions
- Points at bottom = incorrect predictions
- Shows if LLM's confidence is realistic

---

## 🚀 How to Generate These Plots Yourself

### Quick Summary (Text Only)
```bash
python code/analyze_results.py
```

Output:
```
📊 LLM DISCRIMINATION RESULTS SUMMARY
Total trials: 65
Overall accuracy: 23.1%
[Tables showing breakdown by method, condition, label]
```

### Generate Specific Visualization
```bash
# Generate only heatmap
python code/analyze_results.py --plot heatmap

# Generate only bar chart
python code/analyze_results.py --plot accuracy

# Generate only scatter plot
python code/analyze_results.py --plot scatter
```

### Generate ALL Visualizations
```bash
python code/analyze_results.py --plot all
```

**Output:** 5 PNG files saved to `results/plots/`:
- `01_accuracy_by_method.png` — Horizontal bar chart
- `02_heatmap_method_x_condition.png` — Heatmap
- `03_confidence_distribution.png` — Boxplot
- `04_grouped_accuracy.png` — Grouped bars
- `05_confidence_vs_correctness.png` — Scatter plot

---

## 💡 Interpreting Your Current Results

### Overall Performance:
- **23.1% accuracy** = LLM struggling to distinguish synthetic from real
- **Expected baseline** = 50% (random guessing)
- **Your result** = 46% worse than random!

### By Method:
| Method | Accuracy | Interpretation |
|--------|----------|-----------------|
| gaussian_copula | 40% | Best; likely most detectable artifacts |
| ctgan | 27% | Moderate; fairly realistic |
| great | 25% | Good mimicry of real data |
| cart | 12% | Excellent quality; almost indistinguishable |
| tvae | 12% | Excellent quality; almost indistinguishable |

### Condition Effect (C1 vs C2):
- **C1 accuracy**: 21% (hard to detect from table alone)
- **C2 accuracy**: 25% (metadata helps slightly)
- **Marginal improvement**: Only +4% with metadata
- **Interpretation**: LLM struggles even with statistical metadata

### Label Bias:
- **Real accuracy**: 31% (better at identifying real data)
- **Synthetic accuracy**: 15% (worse at identifying synthetic)
- **Interpretation**: LLM has bias toward labeling as "REAL"

---

## 🎯 What This Means for Research

### Positive Findings:
✅ Framework is **fully functional** end-to-end
✅ LLM provides **detailed reasoning** (red flags, evidence)
✅ Framework works across **multiple evaluation conditions**

### Negative Findings:
❌ **Synthetic data is surprisingly hard to detect** (46% worse than chance)
❌ Suggests modern synthetic methods are producing **high-quality, realistic data**
❌ Even with metadata, LLM can't reliably distinguish

### Next Steps:
1. **Increase trials** → 10+ per condition for statistical significance
2. **Try better LLM** → Claude 3.5 Opus (better reasoning)
3. **Analyze reasoning chains** → Why LLM fails on specific methods
4. **Compare methods** → CTGAN vs TVAE vs GaussianCopula in detail
5. **Human baseline** → Compare against human discrimination

---

## 📋 Using analyze_results.py for Different Questions

### "Which method is hardest to detect?"
```bash
python code/analyze_results.py
# Look at "By Synthetic Method" table
# Lowest accuracy = hardest to detect
```

### "Does metadata (C2) help LLM?"
```bash
python code/analyze_results.py --plot heatmap
# Compare C1 vs C2 columns
# If C2 > C1, metadata helps
```

### "Is LLM's confidence justified?"
```bash
python code/analyze_results.py --plot scatter
# Look for correlation between dots at top (correct) vs bottom (wrong)
# High correlation = confidence is well-calibrated
```

### "Which condition×method combo performs best?"
```bash
python code/analyze_results.py
# Look at detailed breakdown table
# Find highest mean value
```

---

## 🔧 Customization

### Analyze Only Specific Provider:
```bash
python code/analyze_results.py --pattern google
# Only analyze Google Gemini results
```

### Analyze Only Specific Method:
```bash
python code/analyze_results.py --pattern ctgan
# Only analyze CTGAN trials
```

### Save Plots to Custom Location:
```bash
python code/analyze_results.py --plot all --output my_results/
# Saves to my_results/ instead of results/plots/
```

---

## 📚 References

For more info on your experiment:
- See `code/evaluate.py` — How trials are executed
- See `code/prompt_builder.py` — How prompts are constructed  
- See `results/llm_results_*.jsonl` — Raw trial data with reasoning
- See `FREE_API_SETUP.md` — How to set up free APIs

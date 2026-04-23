# LLM as Discriminator — Implementation Guide

## Status

### ✓ Complete (Phase 1 & 2)
- [x] Data directory structure created
- [x] Adult dataset downloaded (30,162 rows × 15 columns)
- [x] Dummy synthetic datasets created for all 5 methods (ctgan, tvae, cart, great, tabddpm)
- [x] Dataset inventory created (`datasets_inventory.json`)
- [x] `evaluate.py` extended for multi-method testing
- [x] CLI supports both legacy single-method and new multi-method modes

### ⏰ Next Steps

#### 1. **Replace dummy data with YOUR synthetic datasets**
   - Location: `data/synthetic/{method_name}/synthetic.csv`
   - Each method folder should contain a CSV matching the shape of the real data (30,162 × 15)
   - Methods: `ctgan`, `tvae`, `cart`, `great`, `tabddpm`, etc.
   - Script to help: `python test_multi_method.py` (regenerates CSV structure)

#### 2. **Run pilot LLM trials (small scale, low cost)**
   ```bash
   # Quick test with just 2 trials per condition (8 total per method)
   python code/evaluate.py --real data/real/adult.csv \
     --provider openai \
     --model gpt-4o \
     --n_trials 2 \
     --conditions C1 C2
   ```
   - This will run: 2 methods × 2 conditions × 2 trials = 8 API calls per method
   - Check results in `results/llm_results_*.jsonl`
   - Review reasoning field in first few results to assess prompt quality

#### 3. **Extended trials with both providers**
   ```bash
   # Run 5 trials per condition on OpenAI
   python code/evaluate.py --real data/real/adult.csv \
     --provider openai \
     --model gpt-4o \
     --n_trials 5 \
     --conditions C1 C2
   
   # Then run same on Anthropic
   python code/evaluate.py --real data/real/adult.csv \
     --provider anthropic \
     --model claude-opus-4-5-20251101 \
     --n_trials 5 \
     --conditions C1 C2
   ```

#### 4. **Analyze results**
   - Load all `results/*.jsonl` files
   - Compare accuracy by method, model, condition
   - See [code/analysis.ipynb](code/analysis.ipynb) (to be expanded)

#### 5. **Integrate Slido for human baseline** (optional, pending)
   - Current Streamlit interface in [code/app.py](code/app.py)
   - Plan: Embed Slido poll or native Streamlit voting

---

## Project Structure

```
data/
├── real/
│   └── adult.csv                    ✓ (30,162 rows)
└── synthetic/
    ├── ctgan/synthetic.csv         ⚠ (dummy, replace)
    ├── tvae/synthetic.csv          ⚠ (dummy, replace)
    ├── cart/synthetic.csv          ⚠ (dummy, replace)
    ├── great/synthetic.csv         ⚠ (dummy, replace)
    └── tabddpm/synthetic.csv       ⚠ (dummy, replace)

code/
├── prepare_data.py                  ✓ Data download & organization
├── prompt_builder.py                ✓ C1/C2 prompt generation
├── metadata.py                      ✓ Statistical metadata extraction
├── evaluate.py                      ✓ Multi-method LLM trials
├── app.py                          (Streamlit human interface)
├── analysis.ipynb                  (To be expanded)
└── environment.yml                 (Dependencies)

results/
└── llm_results_*.jsonl             (Generated after running trials)

datasets_inventory.json             ✓ Created (dataset metadata)
```

---

## Key Files & Their Roles

| File | Purpose | Status |
|------|---------|--------|
| `code/prepare_data.py` | Download Adult, validate synthetic structure | ✓ Ready |
| `code/evaluate.py` | Run LLM trials across methods | ✓ Extended |
| `code/prompt_builder.py` | Build C1/C2 prompts | ✓ Ready (may refine) |
| `code/metadata.py` | Extract rich statistical metadata | ✓ Ready |
| `code/app.py` | Streamlit human interface | ⏳ Pending Slido integration |
| `code/analysis.ipynb` | Results analysis & visualization | ⏳ To be built |

---

## CLI Commands Reference

### Multi-method (auto-discovers all methods)
```bash
python code/evaluate.py --real data/real/adult.csv \
  --provider openai \
  --model gpt-4o \
  --n_trials 5
```

### Specific methods only
```bash
python code/evaluate.py --real data/real/adult.csv \
  --methods ctgan tvae cart \
  --provider openai \
  --n_trials 5
```

### Legacy single-method mode
```bash
python code/evaluate.py --real data/real/adult.csv \
  --synthetic data/synthetic/ctgan/synthetic.csv \
  --provider openai \
  --n_trials 5
```

### Full options
```
--real PATH                Real data CSV (required)
--synthetic PATH           Synthetic CSV (legacy single-method)
--synthetic-dir DIR        Synthetic methods directory (default: data/synthetic)
--methods NAME [NAME ...]  Specific methods (default: auto-discover)
--provider {openai,anthropic}  LLM provider (default: openai)
--model MODEL             Model name (default: gpt-4o)
--n_trials N              Trials per label per condition (default: 10)
--conditions C1 C2        Experimental conditions (default: C1 C2)
--n_rows_shown N          Rows of data shown in prompt (default: 20)
--sample_rows N           Sub-sample data per trial (default: None)
```

---

## Estimated Costs & Time

### Pilot run (2 trials, 1 method, 1 provider)
- **Trials**: 8 (2 conditions × 2 labels × 2 trials)
- **Time**: ~4 min
- **Cost**: ~$0.05 (GPT-4o) or free tier (Anthropic)

### Full run (5 trials, 5 methods, 2 providers each)
- **Trials**: 200 total (5 methods × 2 conditions × 5 trials × 2 providers)
- **Time**: ~2 hours
- **Cost**: ~$5-10 (depends on model + rate limiting)

---

## Next Immediate Actions

1. **Replace dummy data** with your actual synthetic datasets
   - If you have pre-generated synthetic CSVs, copy them to `data/synthetic/{method}/synthetic.csv`
   - Ensure same column names and shape as real data

2. **Set environment variables for LLM APIs**
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

3. **Run pilot trial** to verify prompts & check for errors
   ```bash
   python code/evaluate.py --real data/real/adult.csv --n_trials 2 --provider openai
   ```

4. **Review results** in `results/llm_results_*.jsonl`
   - Check confidence scores and reasoning field
   - Assess if LLM is making meaningful distinctions

5. **Iterate & scale up** based on pilot results

---

## Questions?

- **Do you have synthetic CSVs ready?** If so, share their paths and I'll help organize them.
- **Which LLM models want to test?** Current setup: GPT-4o (OpenAI) + Claude Opus (Anthropic)
- **Human baseline timing?** When do you want to set up the Slido voting?

# LLM vs Human Discriminator — Real or Synthetic Tabular Data

A research framework to evaluate whether LLMs and humans can distinguish real
from synthetic tabular data, across two information conditions.

---

## Project Structure

```
llm_discriminator/
├── data/
│   ├── real/           ← place your real.csv here
│   └── synthetic/      ← place your synthetic.csv here
├── llm_discriminator/
│   ├── metadata.py     ← derives distributional metadata from any dataframe
│   ├── prompt_builder.py  ← constructs C1 / C2 prompts
│   └── evaluate.py     ← runs LLM discrimination trials (CLI)
├── human_interface/
│   └── app.py          ← Streamlit labeling UI for human participants
├── analysis/
│   └── analysis.ipynb  ← full results analysis notebook
├── results/            ← auto-created; all .jsonl and .csv outputs land here
├── envs/
│   ├── environment.yml ← conda env for SURF/DAS
│   └── requirements.txt
├── .env.example        ← copy to .env and add your API keys
└── README.md
```

---

## Setup

### 1. Environment (recommended: conda on SURF/DAS)

```bash
# Clone / copy the project
cd llm_discriminator

# Create conda environment
conda env create -f envs/environment.yml
conda activate llm_discriminator

# OR use pip only
python -m venv .venv
source .venv/bin/activate
pip install -r envs/requirements.txt
```

### 2. API Keys

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

### 3. Add your data

```bash
cp /path/to/your/real_data.csv       data/real/real.csv
cp /path/to/your/synthetic_data.csv  data/synthetic/synthetic.csv
```

---

## Running the LLM Discriminator

```bash
# Basic run — OpenAI GPT-4o, both conditions, 10 trials each
python -m llm_discriminator.evaluate \
    --real data/real/real.csv \
    --synthetic data/synthetic/synthetic.csv \
    --provider openai \
    --model gpt-4o \
    --n_trials 10 \
    --conditions C1 C2

# Anthropic Claude
python -m llm_discriminator.evaluate \
    --real data/real/real.csv \
    --synthetic data/synthetic/synthetic.csv \
    --provider anthropic \
    --model claude-opus-4-5-20251101 \
    --n_trials 10
```

Results are written to `results/llm_results_<provider>_<timestamp>.jsonl`

---

## Running the Human Interface

```bash
streamlit run human_interface/app.py
```

- Opens at http://localhost:8501
- Participants enter their ID, choose conditions, and judge tables one by one
- Results are appended to `results/human_results.jsonl` in real time

### Deploying on SURF/DAS (port forwarding)

```bash
# On the compute node:
streamlit run human_interface/app.py --server.port 8501 --server.address 0.0.0.0

# On your local machine (SSH tunnel):
ssh -L 8501:localhost:8501 <username>@<surf-node-address>
# Then open http://localhost:8501 in your browser
```

---

## Running the Analysis Notebook

```bash
cd analysis
jupyter notebook analysis.ipynb
# OR on SURF/DAS:
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_executed.ipynb
```

---

## Experimental Conditions

| Condition | What the evaluator sees |
|-----------|------------------------|
| **C1**    | Raw table (first N rows) |
| **C2**    | Raw table + full distributional metadata (mean, std, skewness, kurtosis, top values, correlations, Shapiro-Wilk normality, etc.) |

---

## Output Files

| File | Description |
|------|-------------|
| `results/llm_results_<provider>_<ts>.jsonl` | LLM trial-level results |
| `results/human_results.jsonl` | Human trial-level results (all participants) |
| `results/final_summary.csv` | Aggregated accuracy + significance tests |
| `results/*.png` | Analysis plots |

---

## Extending the Study

- **Add a synthesizer:** Generate a new synthetic CSV with SDV (`sdv.SingleTableMetadata` + any synthesizer) and drop it in `data/synthetic/`
- **Add a model:** Add a new entry to `PROVIDER_MAP` in `evaluate.py`
- **Add a condition C3:** Extend `prompt_builder.py` with a new template and add `"C3"` to the CLI options
- **Multi-dataset:** Loop `run_experiment()` over multiple `(real_path, synthetic_path)` pairs

---

## Citation (when publishing)

If you use this framework, please cite the relevant synthesizer papers
(CTGAN, GReaT, REaLTabFormer) and the LLM-as-evaluator literature
(Zheng et al. 2023).

# LLM-as-a-Discriminator

Practical runbook for reproducing experiments and analyses in this repository.

## 1) Environment Setup

Choose one option.

### Option A: Conda (recommended)
```bash
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
conda env create -f code/environment.yml
conda activate psd2026
```

### Option B: venv + pip
```bash
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy matplotlib seaborn scikit-learn scipy python-dotenv google-generativeai groq sdv jupyter streamlit
```

## 2) API Keys

Create `.env` in the project root:

```bash
cat > .env << 'EOF'
GOOGLE_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
EOF
```

Notes:
- `.env` is ignored by git.
- Never commit secrets.

## 3) Data Preparation

### Adult dataset (if needed)
```bash
python generate_synthetic_data.py
```

### ACS Census data
```bash
python code/prepare_census_data.py
```

## 4) Run LLM Evaluation

### Single run (manual)
```bash
python code/evaluate.py \
  --real data/real/adult.csv \
  --provider google \
  --model gemini-2.5-flash \
  --n_trials 50
```

### Resume larger batch (existing helper)
```bash
bash resume_phase2.sh
```

### Census experiment (helper scripts)
```bash
bash run_census_experiment.sh
# or
bash run_census_background.sh
```

## 5) Analysis and Figures

Run available analysis scripts:

```bash
python analyze_results.py
python analyze_census.py
python analyze_reasoning.py
python code/condition_analysis.py
python code/formal_privacy_measurement.py
python make_final_plots.py
```

Main outputs are written under `results/`.

## 6) Human Labeling App (optional)

```bash
streamlit run human_labeling_app.py
```

## 7) Notebook Workflow (optional)

```bash
jupyter notebook code/analysis_results.ipynb
```

## 8) Paper Build (optional)

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## 9) Minimal Repro Pipeline

If you want the shortest end-to-end path:

```bash
source .venv/bin/activate
python code/prepare_census_data.py
bash run_census_experiment.sh
python make_final_plots.py
```

## License

MIT (see LICENSE).

## Contact
Manel Slokom: manel.slokom@live.fr

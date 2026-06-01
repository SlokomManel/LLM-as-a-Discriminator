#!/usr/bin/env bash
# =============================================================================
# run_census_experiment.sh
# Full pipeline for the US Census (ACS Income) dataset.
# Adult results are NEVER touched — all census outputs go to separate folders.
#
# Run from the project root:
#   chmod +x run_census_experiment.sh
#   ./run_census_experiment.sh            # fresh run
#   ./run_census_experiment.sh --resume   # skip already-completed trials
#
# Prerequisites:
#   source .venv/bin/activate
#   pip install folktables        # only needed once
#   # Set your LLM API key(s) in .env
# =============================================================================
set -euo pipefail

RESUME_FLAG=""
for arg in "$@"; do
    if [[ "$arg" == "--resume" ]]; then
        RESUME_FLAG="--resume"
    fi
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

source .venv/bin/activate

# ── Configurable parameters ───────────────────────────────────────────────────
YEAR=2018
# STATE="CA"      # uncomment to restrict to one state (~32 K rows, faster)
N_SAMPLES=32561   # match Adult dataset size for comparability (48842 -> ~32 K)
REAL_DATA="data/census/real/census.csv"
SYNTH_DIR="data/census/synthetic"
RESULTS_DIR="results/census"

# LLM experiment settings — matching Adult dataset setup exactly
# Google: gemini-2.5-flash | Groq: llama-3.1-8b-instant
N_TRIALS=10   # trials per label per condition per method (total = n_trials*2*2*3 = 120 per model)

# =============================================================================
echo ""
echo "============================================================"
echo " CENSUS EXPERIMENT PIPELINE"
echo " Real data:    $REAL_DATA"
echo " Synthetic:    $SYNTH_DIR"
echo " Results:      $RESULTS_DIR"
echo "============================================================"
echo ""

# ── Step 1: Download / prepare census data ────────────────────────────────────
if [[ -f "$REAL_DATA" ]]; then
    echo "[SKIP] Census data already exists at $REAL_DATA"
else
    echo "[STEP 1] Downloading ACS $YEAR data..."
    python code/prepare_census_data.py \
        --year "$YEAR" \
        --n_samples "$N_SAMPLES" \
        --output "$REAL_DATA"
    echo "[DONE] Census data saved to $REAL_DATA"
fi

# ── Step 2: Generate synthetic data ───────────────────────────────────────────
echo ""
SYNTH_MISSING=0
for METHOD in ctgan tvae gaussian_copula; do
    [[ ! -f "$SYNTH_DIR/$METHOD/synthetic.csv" ]] && SYNTH_MISSING=1
done

if [[ "$SYNTH_MISSING" -eq 0 ]]; then
    echo "[SKIP] All synthetic datasets already exist in $SYNTH_DIR/"
else
    echo "[STEP 2] Generating synthetic census data (CTGAN, TVAE, GaussianCopula)..."
    python generate_synthetic_data.py \
        --real "$REAL_DATA" \
        --output_dir "$SYNTH_DIR" \
        --ctgan_epochs 100 \
        --tvae_epochs 100
    echo "[DONE] Synthetic data written to $SYNTH_DIR/{ctgan,tvae,gaussian_copula}/synthetic.csv"
fi

# ── Step 3: Run LLM discrimination trials ─────────────────────────────────────
echo ""
echo "[STEP 3a] Running Google Gemini 2.5 Flash trials (matches Adult experiment)..."
python code/evaluate.py \
    --real "$REAL_DATA" \
    --synthetic-dir "$SYNTH_DIR" \
    --results-dir "$RESULTS_DIR" \
    --provider google \
    --model gemini-2.5-flash \
    --n_trials "$N_TRIALS" \
    $RESUME_FLAG
echo "[DONE] Google results written to $RESULTS_DIR/llm_results_*_google_*.jsonl"

echo ""
echo "[STEP 3b] Running Groq Llama-3.1-8b trials (matches Adult experiment)..."
python code/evaluate.py \
    --real "$REAL_DATA" \
    --synthetic-dir "$SYNTH_DIR" \
    --results-dir "$RESULTS_DIR" \
    --provider groq \
    --model llama-3.1-8b-instant \
    --n_trials "$N_TRIALS" \
    $RESUME_FLAG
echo "[DONE] Groq results written to $RESULTS_DIR/llm_results_*_groq_*.jsonl"

# ── Step 4: Formal privacy measurement ────────────────────────────────────────
echo ""
echo "[STEP 4] Computing Record Linkage Risk + MIA for census data..."
python code/formal_privacy_measurement.py \
    --real "$REAL_DATA" \
    --synthetic-dir "$SYNTH_DIR" \
    --results-dir "$RESULTS_DIR" \
    --llm-results-dir "$RESULTS_DIR"
echo "[DONE] Privacy measurements written to $RESULTS_DIR/"

echo ""
echo "============================================================"
echo " ALL STEPS COMPLETE"
echo " Results in: $RESULTS_DIR/"
echo " Adult results are untouched in: results/"
echo "============================================================"

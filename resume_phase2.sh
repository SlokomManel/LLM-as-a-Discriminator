#!/bin/bash
# resume_phase2.sh
# Resume Phase 2 multi-model testing TOMORROW after midnight UTC quota reset
# This runs ONLY the remaining models that haven't been fully tested yet

echo "🔄 Phase 2 Resume Script"
echo "======================================"
echo "Remaining models to test:"
echo "  1. gemini-2.0-flash"
echo "  2. gemini-3-flash-preview"  
echo "  3. gemini-3-pro-preview"
echo "  4. gemma-3-27b-it"
echo ""
echo "Total expected: 4 models × 6 methods × 2 conditions × 50 trials = ~2,400 new trials"
echo "======================================"
echo ""

# Ensure API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ ERROR: GOOGLE_API_KEY not set"
    echo "Set it with: export GOOGLE_API_KEY='your-key-here'"
    exit 1
fi

# Activate venv
cd /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026
source .venv/bin/activate

# Run the remaining models in background with nohup
# Each model will take ~15-20 minutes across all 6 synthetic methods
echo "🚀 Starting Phase 2 resume (models: gemini-2.0-flash, gemini-3-flash-preview, gemini-3-pro-preview, gemma-3-27b-it)..."
echo ""

nohup python code/evaluate.py \
  --real data/real/adult.csv \
  --provider google \
  --compare-models \
    gemini-2.0-flash \
    gemini-3-flash-preview \
    gemini-3-pro-preview \
    gemma-3-27b-it \
  --n_trials 50 \
  > experiment_phase2_resume.log 2>&1 &

PID=$!
echo "✅ Process started with PID: $PID"
echo "📊 Check progress with: tail -f experiment_phase2_resume.log"
echo "📈 Count results with: wc -l results/llm_results_*.jsonl"

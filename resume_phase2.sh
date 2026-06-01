#!/bin/bash
# resume_phase2.sh
# Resume Phase 2 LLM discrimination testing with free/included models
# Runs sequentially in background - you can close terminal/laptop
# Automatically continues from where it stopped when tokens are exhausted

WORK_DIR="/Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026"
LOG_FILE="$WORK_DIR/experiment_phase2_resume.log"

echo "🔄 Phase 2 LLM Discrimination - Multi-Provider Testing (Free Models)"
echo "======================================"
echo "Providers & Models:"
echo ""
echo "  🔵 GOOGLE (Gemini) - FREE tier:"
echo "    1. gemini-2.5-flash"
echo "    2. gemini-2.0-flash"
echo "    3. gemini-3-flash-preview"  
echo "    4. gemini-3-pro-preview"
echo ""
echo "  🟢 GROQ (OSS Models) - FREE tier:"
echo "    5. llama-3.1-8b-instant       (8B, fastest)"
echo "    6. gemma-3-27b-it             (27B, instruction-tuned)"
echo "    7. mixtral-8x7b-32768         (47B, mixture-of-experts)"
echo "    8. llama-3.1-70b-versatile    (70B, most capable)"
echo ""
echo "  🟠 GITHUB MODELS (via Copilot) - OPTIONAL:"
echo "    9. gpt-4o (if GITHUB_PAT set)"
echo ""
echo "Structure: 4 (Google) + 4 (Groq) models × 3 methods × 2 conditions × 50 trials"
echo "Total expected: ~2,400 new trials (Google) + ~2,400 new trials (Groq) = ~4,800 trials"
echo "======================================"
echo ""

# Check for required API keys
echo "🔑 Checking API keys..."
MISSING_KEYS=()

if [ -z "$GOOGLE_API_KEY" ]; then
    MISSING_KEYS+=("GOOGLE_API_KEY")
fi
if [ -z "$GROQ_API_KEY" ]; then
    MISSING_KEYS+=("GROQ_API_KEY")
fi

if [ ${#MISSING_KEYS[@]} -gt 0 ]; then
    echo "❌ REQUIRED API keys missing:"
    for key in "${MISSING_KEYS[@]}"; do
        echo "   - $key"
    done
    echo ""
    echo "Please set them with: export KEY='your-key'"
    exit 1
fi

# Optional GitHub Models
if [ -z "$GITHUB_PAT" ]; then
    echo "⚠️  GITHUB_PAT not set - skipping GitHub Models (optional)"
else
    echo "✅ GITHUB_PAT detected - will test GitHub Models"
fi

echo ""

# Check current result count
CURRENT_COUNT=$(find "$WORK_DIR/results/" -name "llm_results_*.jsonl" -type f 2>/dev/null | wc -l)
echo "📊 Current result files: $CURRENT_COUNT"
echo ""

# Run experiments in background (will append to existing files if they exist)
# This naturally supports resuming after token exhaustion
echo "🚀 Starting multi-provider experiments in background..."
echo "   Output: $LOG_FILE"
echo ""

nohup bash -c "
  cd '$WORK_DIR'
  source .venv/bin/activate
  
  echo '=====================================================' >> '$LOG_FILE'
  echo '📊 Starting Multi-Provider LLM Discrimination at '$(date)'' >> '$LOG_FILE'
  echo '=====================================================' >> '$LOG_FILE'
  
  # GOOGLE PROVIDER (Required)
  echo '' >> '$LOG_FILE'
  echo '🔵 GOOGLE PROVIDER - Testing Gemini models' >> '$LOG_FILE'
  export GOOGLE_API_KEY='$GOOGLE_API_KEY'
  
  python code/evaluate.py \
    --real data/real/adult.csv \
    --provider google \
    --compare-models \
      gemini-2.5-flash \
      gemini-2.0-flash \
      gemini-3-flash-preview \
      gemini-3-pro-preview \
    --n_trials 50 >> '$LOG_FILE' 2>&1
  
  echo '✅ Google provider complete' >> '$LOG_FILE'
  
  # GROQ PROVIDER (Required)
  echo '' >> '$LOG_FILE'
  echo '🟢 GROQ PROVIDER - Testing diverse OSS models' >> '$LOG_FILE'
  export GROQ_API_KEY='$GROQ_API_KEY'
  
  python code/evaluate.py \
    --real data/real/adult.csv \
    --provider groq \
    --compare-models \
      llama-3.1-8b-instant \
      gemma-3-27b-it \
      mixtral-8x7b-32768 \
      llama-3.1-70b-versatile \
    --n_trials 50 >> '$LOG_FILE' 2>&1
  
  echo '✅ Groq provider complete' >> '$LOG_FILE'
  
  # GITHUB MODELS PROVIDER (Optional - if GITHUB_PAT set)
  if [ ! -z '$GITHUB_PAT' ]; then
    echo '' >> '$LOG_FILE'
    echo '🟠 GITHUB MODELS PROVIDER - Testing via Copilot' >> '$LOG_FILE'
    export GITHUB_PAT='$GITHUB_PAT'
    
    python code/evaluate.py \
      --real data/real/adult.csv \
      --provider github \
      --compare-models gpt-4o \
      --n_trials 50 >> '$LOG_FILE' 2>&1
    
    echo '✅ GitHub Models provider complete' >> '$LOG_FILE'
  else
    echo '⏭️  GITHUB MODELS PROVIDER - Skipped (GITHUB_PAT not set)' >> '$LOG_FILE'
  fi
  
  echo '' >> '$LOG_FILE'
  echo '✅ All experiments complete at '$(date)'' >> '$LOG_FILE'
  echo '📊 Running analysis...' >> '$LOG_FILE'
  
  python code/analyze_results.py >> '$LOG_FILE' 2>&1
  
  echo '📈 Analysis complete. Results saved in results/ folder.' >> '$LOG_FILE'
  echo '=====================================================' >> '$LOG_FILE'
" > /dev/null 2>&1 &

PID=$!
echo "✅ Background process started (PID: $PID)"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📌 MONITORING:"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1️⃣  Watch progress in real-time:"
echo "   tail -f $LOG_FILE"
echo ""
echo "2️⃣  Count current results:"
echo "   wc -l results/llm_results_*.jsonl | tail -1"
echo ""
echo "3️⃣  Check if process still running:"
echo "   ps aux | grep evaluate.py"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "💡 RESUMING AFTER TOKEN EXHAUSTION:"
echo ""
echo "   Just run this script again after token reset:"
echo "   bash resume_phase2.sh"
echo ""
echo "   Results will automatically append to existing files!"
echo "═══════════════════════════════════════════════════════════════"

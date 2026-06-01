#!/usr/bin/env bash
# run_census_background.sh
#
# Runs the census experiment inside a persistent tmux session so it survives
# terminal disconnection.  The evaluate.py script uses quota-aware back-off
# (up to 10 min between retries) and can be restarted with --resume to skip
# already-completed trials.
#
# Usage:
#   ./run_census_background.sh            # start fresh experiment
#   ./run_census_background.sh --resume   # resume after quota killed the run
#
# tmux commands:
#   tmux attach -t census                 # reattach to watch progress
#   tmux kill-session -t census           # stop the session

set -euo pipefail

SESSION="census"
VENV="$(pwd)/.venv/bin/activate"
RESUME="${1:-}"

# ── Config ────────────────────────────────────────────────────────────────────
YEAR=2018
N_SAMPLES=32561          # match Adult dataset size; remove to use all ~195k rows
REAL_DATA="data/census/real/census.csv"
SYNTH_DIR="data/census/synthetic"
RESULTS_DIR="results/census"
PROVIDER="google"
MODEL="gemini-2.5-flash"
N_TRIALS=10
# ─────────────────────────────────────────────────────────────────────────────

if ! command -v tmux &>/dev/null; then
    echo "tmux not found. Install it first: brew install tmux"
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists."
    echo "  Attach with: tmux attach -t $SESSION"
    echo "  Kill with:   tmux kill-session -t $SESSION"
    exit 0
fi

# Always resume — safe on first run (no files yet = starts fresh automatically)
CMD="bash run_census_experiment.sh --resume"

echo "Starting tmux session '$SESSION'..."
echo "  Attach with: tmux attach -t $SESSION"
echo "  Detach with: Ctrl-B then D"

tmux new-session -d -s "$SESSION" -x 220 -y 50 \
    "source $VENV && $CMD; echo ''; echo '=== DONE (exit \\$?) ==='; bash"

echo "Session started. Run 'tmux attach -t $SESSION' to watch."

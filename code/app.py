"""
app.py  —  Human Discrimination Interface
Run with: streamlit run human_interface/app.py

The interface presents one table at a time (real or synthetic, randomly selected)
and asks the human to judge whether it is REAL or SYNTHETIC.
Conditions: C1 (table only) or C2 (table + metadata).
Results are saved to results/human_results.jsonl
"""

import os
import json
import random
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Allow imports from current directory
sys.path.insert(0, str(Path(__file__).parent))
from metadata import extract_metadata, metadata_to_text

load_dotenv()

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HUMAN_RESULTS_FILE = RESULTS_DIR / "human_results.jsonl"

REAL_PATH = os.getenv("REAL_DATA_PATH", "data/real/real.csv")
SYNTHETIC_PATH = os.getenv("SYNTHETIC_DATA_PATH", "data/synthetic/synthetic.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real or Synthetic?",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0d0f14;
    color: #e8e6e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #2a2d38;
}

/* Main header */
.main-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f0e6c8 0%, #c8a96e 50%, #f0e6c8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.sub-header {
    color: #6b7280;
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Trial card */
.trial-card {
    background: #13161e;
    border: 1px solid #2a2d38;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Condition badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.badge-c1 { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
.badge-c2 { background: #2d1e5f; color: #c084fc; border: 1px solid #7c3aed; }

/* Table styling */
.stDataFrame { font-family: 'DM Mono', monospace; font-size: 0.8rem; }

/* Metadata block */
.metadata-block {
    background: #0a0c10;
    border: 1px solid #2a2d38;
    border-left: 3px solid #c8a96e;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #9ca3af;
    white-space: pre-wrap;
    margin-top: 1rem;
    max-height: 400px;
    overflow-y: auto;
}

/* Vote buttons */
.stButton > button {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    transition: all 0.2s;
    width: 100%;
}

/* Progress */
.progress-text {
    color: #6b7280;
    font-size: 0.85rem;
    font-family: 'DM Mono', monospace;
}

/* Score display */
.score-card {
    background: #13161e;
    border: 1px solid #2a2d38;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.score-number {
    font-size: 2.5rem;
    font-weight: 800;
    color: #c8a96e;
}

/* Verdict feedback */
.verdict-correct {
    background: #052e16;
    border: 1px solid #16a34a;
    color: #4ade80;
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
}
.verdict-wrong {
    background: #2d0a0a;
    border: 1px solid #dc2626;
    color: #f87171;
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    real_df = pd.read_csv(REAL_PATH)
    synthetic_df = pd.read_csv(SYNTHETIC_PATH)
    return real_df, synthetic_df


@st.cache_data
def get_metadata(df_hash, df):
    return extract_metadata(df)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "trial_id": 0,
        "results": [],
        "current_label": None,
        "current_condition": None,
        "current_df": None,
        "show_verdict": False,
        "last_correct": None,
        "session_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "participant_id": None,
        "started": False,
        "n_trials": 10,
        "conditions": ["C1", "C2"],
        "n_rows_shown": 15,
        "confidence": 50,
        "queue": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Sidebar: Setup ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Study Settings")

    if not st.session_state.started:
        participant_id = st.text_input(
            "Your participant ID",
            placeholder="e.g. P001",
            help="Enter a unique ID for your session.",
        )
        n_trials = st.slider("Trials per condition", 4, 30, 10, 2)
        conditions = st.multiselect(
            "Active conditions",
            ["C1", "C2"],
            default=["C1", "C2"],
            help="C1 = table only | C2 = table + metadata",
        )
        n_rows_shown = st.slider("Rows shown per table", 5, 50, 15)

        st.markdown("---")
        st.markdown("**Data paths**")
        st.code(f"Real: {REAL_PATH}\nSynthetic: {SYNTHETIC_PATH}", language=None)

        if st.button("🚀 Start Study", use_container_width=True):
            if not participant_id:
                st.error("Please enter a participant ID.")
            else:
                st.session_state.participant_id = participant_id
                st.session_state.n_trials = n_trials
                st.session_state.conditions = conditions
                st.session_state.n_rows_shown = n_rows_shown

                # Build balanced, shuffled trial queue
                queue = []
                for cond in conditions:
                    for _ in range(n_trials // 2):
                        queue.append(("REAL", cond))
                        queue.append(("SYNTHETIC", cond))
                random.shuffle(queue)
                st.session_state.queue = queue
                st.session_state.started = True
                st.rerun()
    else:
        st.markdown(f"**Participant:** `{st.session_state.participant_id}`")
        st.markdown(f"**Session:** `{st.session_state.session_id}`")
        st.markdown(f"**Conditions:** {', '.join(st.session_state.conditions)}")
        st.markdown("---")

        # Live score
        results = st.session_state.results
        if results:
            correct = sum(r["correct"] for r in results)
            total = len(results)
            st.markdown(f"**Score:** {correct}/{total} ({100*correct/total:.0f}%)")

            # Per-condition breakdown
            for cond in st.session_state.conditions:
                cond_results = [r for r in results if r["condition"] == cond]
                if cond_results:
                    c_correct = sum(r["correct"] for r in cond_results)
                    st.markdown(f"  • {cond}: {c_correct}/{len(cond_results)}")

        st.markdown("---")
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Real or Synthetic?</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Tabular Data Discrimination Study</div>', unsafe_allow_html=True)

if not st.session_state.started:
    # Landing page
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="score-card">
            <div style="font-size:2rem">🔍</div>
            <div style="color:#9ca3af; font-size:0.85rem; margin-top:0.5rem">Examine the data carefully</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="score-card">
            <div style="font-size:2rem">🧠</div>
            <div style="color:#9ca3af; font-size:0.85rem; margin-top:0.5rem">Use your intuition and expertise</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="score-card">
            <div style="font-size:2rem">📊</div>
            <div style="color:#9ca3af; font-size:0.85rem; margin-top:0.5rem">Judge: real or machine-generated?</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    ---
    **Instructions:** You will be shown a series of tabular datasets. Some are real data collected from real-world observations. 
    Others are **synthetic** — generated by machine learning models designed to mimic real data distributions.

    Your task is to decide for each table whether it is **REAL** or **SYNTHETIC**, 
    and rate your confidence from 0 (complete guess) to 100 (certain).

    👈 Configure your session in the sidebar and click **Start Study** to begin.
    """)
    st.stop()

# ── Active trial ──────────────────────────────────────────────────────────────
real_df, synthetic_df = load_data()
queue = st.session_state.queue
trial_id = st.session_state.trial_id
total_trials = len(queue)

# Study complete
if trial_id >= total_trials:
    st.success("🎉 Study complete! Thank you for your participation.")
    results = st.session_state.results
    if results:
        res_df = pd.DataFrame(results)
        correct = res_df["correct"].sum()
        total = len(res_df)
        st.markdown(f"### Final Score: **{correct}/{total}** ({100*correct/total:.1f}%)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Results by condition:**")
            st.dataframe(
                res_df.groupby("condition")["correct"].agg(["sum", "count", "mean"])
                .rename(columns={"sum": "Correct", "count": "Total", "mean": "Accuracy"})
                .round(3)
            )
        with col2:
            st.markdown("**Results by true label:**")
            st.dataframe(
                res_df.groupby("true_label")["correct"].agg(["sum", "count", "mean"])
                .rename(columns={"sum": "Correct", "count": "Total", "mean": "Accuracy"})
                .round(3)
            )

        st.markdown("**Your full response log:**")
        st.dataframe(res_df[["trial_id", "condition", "true_label", "predicted_label", "correct", "confidence_rating"]])

    st.stop()

# ── Load or prepare current trial ────────────────────────────────────────────
if st.session_state.current_df is None:
    true_label, condition = queue[trial_id]
    df = real_df.copy() if true_label == "REAL" else synthetic_df.copy()
    # Sample rows for variety
    if len(df) > st.session_state.n_rows_shown:
        df_shown = df.sample(n=st.session_state.n_rows_shown, random_state=trial_id).reset_index(drop=True)
    else:
        df_shown = df.reset_index(drop=True)

    st.session_state.current_label = true_label
    st.session_state.current_condition = condition
    st.session_state.current_df = df
    st.session_state.current_df_shown = df_shown

true_label = st.session_state.current_label
condition = st.session_state.current_condition
df = st.session_state.current_df
df_shown = st.session_state.current_df_shown

# ── Progress bar ──────────────────────────────────────────────────────────────
progress_pct = trial_id / total_trials
st.progress(progress_pct)
st.markdown(
    f'<div class="progress-text">Trial {trial_id + 1} of {total_trials} &nbsp;|&nbsp; '
    f'Condition: <strong>{condition}</strong></div>',
    unsafe_allow_html=True,
)
st.markdown("")

# ── Show verdict feedback ─────────────────────────────────────────────────────
if st.session_state.show_verdict:
    last = st.session_state.results[-1]
    if last["correct"]:
        st.markdown(
            f'<div class="verdict-correct">✅ Correct! It was <strong>{last["true_label"]}</strong>.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="verdict-wrong">❌ Wrong. It was <strong>{last["true_label"]}</strong>, you said <strong>{last["predicted_label"]}</strong>.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("")
    if st.button("➡️ Next trial", use_container_width=False):
        st.session_state.current_df = None
        st.session_state.show_verdict = False
        st.rerun()
    st.stop()

# ── Condition badge ───────────────────────────────────────────────────────────
badge_class = "badge-c1" if condition == "C1" else "badge-c2"
badge_label = "Condition 1 — Table only" if condition == "C1" else "Condition 2 — Table + Metadata"
st.markdown(f'<span class="badge {badge_class}">{badge_label}</span>', unsafe_allow_html=True)

# ── Table display ─────────────────────────────────────────────────────────────
st.markdown("**Dataset**")
st.dataframe(df_shown, use_container_width=True, height=380)
st.markdown(
    f'<div class="progress-text">Showing {len(df_shown)} of {len(df)} rows &nbsp;|&nbsp; '
    f'{df.shape[1]} columns</div>',
    unsafe_allow_html=True,
)

# ── Condition 2: Show metadata ────────────────────────────────────────────────
if condition == "C2":
    st.markdown("")
    with st.expander("📊 Statistical Metadata", expanded=True):
        meta = extract_metadata(df)
        meta_text = metadata_to_text(meta)
        st.markdown(f'<div class="metadata-block">{meta_text}</div>', unsafe_allow_html=True)

# ── Decision UI ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🧐 Your Verdict")

confidence_rating = st.slider(
    "Confidence (0 = pure guess, 100 = certain)",
    0, 100, 50,
    key=f"conf_{trial_id}",
)

reasoning_text = st.text_area(
    "Optional: what made you decide? (helps us understand your reasoning)",
    placeholder="e.g. 'The age and income combination looked unrealistic…'",
    key=f"reason_{trial_id}",
    height=80,
)

col1, col2 = st.columns(2)
with col1:
    if st.button("🌍 REAL", use_container_width=True, key=f"real_{trial_id}"):
        predicted = "REAL"
        correct = predicted == true_label
        result = {
            "trial_id": trial_id,
            "timestamp": datetime.utcnow().isoformat(),
            "participant_id": st.session_state.participant_id,
            "session_id": st.session_state.session_id,
            "condition": condition,
            "true_label": true_label,
            "predicted_label": predicted,
            "correct": correct,
            "confidence_rating": confidence_rating,
            "reasoning": reasoning_text,
        }
        st.session_state.results.append(result)
        with open(HUMAN_RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")
        st.session_state.trial_id += 1
        st.session_state.show_verdict = True
        st.rerun()

with col2:
    if st.button("🤖 SYNTHETIC", use_container_width=True, key=f"synthetic_{trial_id}"):
        predicted = "SYNTHETIC"
        correct = predicted == true_label
        result = {
            "trial_id": trial_id,
            "timestamp": datetime.utcnow().isoformat(),
            "participant_id": st.session_state.participant_id,
            "session_id": st.session_state.session_id,
            "condition": condition,
            "true_label": true_label,
            "predicted_label": predicted,
            "correct": correct,
            "confidence_rating": confidence_rating,
            "reasoning": reasoning_text,
        }
        st.session_state.results.append(result)
        with open(HUMAN_RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")
        st.session_state.trial_id += 1
        st.session_state.show_verdict = True
        st.rerun()

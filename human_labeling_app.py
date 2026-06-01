"""
human_labeling_app.py  —  Human Annotation Interface
=====================================================
Mirrors the LLM discrimination experiment exactly:
  • Same datasets  (Adult / ACS Census)
  • Same synthesis methods  (CTGAN / TVAE / Gaussian Copula)
  • Same two conditions  C1 (table only) → C2 (table + metadata)
  • Same 20-row sampled presentation

Paired design: each trial pair shows the SAME 20-row sample first
under C1 and then under C2, so the annotator can see whether
distributional metadata changes their judgement.

Results are written to:
  results/human/human_results_{annotator_id}_{session_id}.csv

Run from project root:
  streamlit run human_labeling_app.py
"""

import sys, os, json, random, hashlib
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "code"))
from metadata import extract_metadata, metadata_to_text   # noqa: E402

RESULTS_DIR = ROOT / "results" / "human"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "adult": {
        "label": "UCI Adult (Census Income 1994)",
        "real":  ROOT / "data" / "real" / "adult.csv",
        "synthetic": {
            "ctgan":           ROOT / "data" / "synthetic" / "ctgan"           / "synthetic.csv",
            "tvae":            ROOT / "data" / "synthetic" / "tvae"            / "synthetic.csv",
            "gaussian_copula": ROOT / "data" / "synthetic" / "gaussian_copula" / "synthetic.csv",
        },
    },
    "census": {
        "label": "ACS Census Income 2018",
        "real":  ROOT / "data" / "census" / "real" / "census.csv",
        "synthetic": {
            "ctgan":           ROOT / "data" / "census" / "synthetic" / "ctgan"           / "synthetic.csv",
            "tvae":            ROOT / "data" / "census" / "synthetic" / "tvae"            / "synthetic.csv",
            "gaussian_copula": ROOT / "data" / "census" / "synthetic" / "gaussian_copula" / "synthetic.csv",
        },
    },
}

METHOD_LABELS = {
    "ctgan":           "CTGAN",
    "tvae":            "TVAE",
    "gaussian_copula": "Gaussian Copula",
}
N_ROWS = 20           # rows shown per trial — matches LLM experiment exactly

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real or Synthetic? — Human Annotation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d0f14; color: #e8e6e0; }
[data-testid="stSidebar"] { background-color: #13161e; border-right: 1px solid #2a2d38; }
.main-header { font-family:'Syne',sans-serif; font-weight:800; font-size:2.2rem;
  letter-spacing:-0.02em; background:linear-gradient(135deg,#f0e6c8 0%,#c8a96e 50%,#f0e6c8 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.3rem; }
.sub-header { color:#6b7280; font-size:0.85rem; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:1.5rem; }
.badge { display:inline-block; padding:0.2rem 0.8rem; border-radius:20px; font-size:0.72rem;
  font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.8rem; }
.badge-c1 { background:#1e3a5f; color:#60a5fa; border:1px solid #2563eb; }
.badge-c2 { background:#2d1e5f; color:#c084fc; border:1px solid #7c3aed; }
.meta-block { background:#0a0c10; border:1px solid #2a2d38; border-left:3px solid #c8a96e;
  border-radius:8px; padding:1rem 1.2rem; font-family:'DM Mono',monospace; font-size:0.76rem;
  color:#9ca3af; white-space:pre-wrap; max-height:400px; overflow-y:auto; margin-top:1rem; }
.verdict-correct { background:#052e16; border:1px solid #16a34a; color:#4ade80;
  padding:0.7rem 1rem; border-radius:8px; font-weight:700; margin-top:0.5rem; }
.verdict-wrong { background:#2d0a0a; border:1px solid #dc2626; color:#f87171;
  padding:0.7rem 1rem; border-radius:8px; font-weight:700; margin-top:0.5rem; }
.trial-info { color:#6b7280; font-size:0.8rem; font-family:'DM Mono',monospace; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ──────────────────────────────────────────────────────

@st.cache_data
def load_dataset(dataset_key: str, method: str):
    """Load real and synthetic dataframes, compute full-dataset metadata."""
    cfg = DATASETS[dataset_key]
    real_df = pd.read_csv(cfg["real"])
    synth_df = pd.read_csv(cfg["synthetic"][method])
    # Pre-compute metadata from the FULL datasets (same as LLM experiment)
    real_meta_text  = metadata_to_text(extract_metadata(real_df,  label="full_real"))
    synth_meta_text = metadata_to_text(extract_metadata(synth_df, label="full_synthetic"))
    return real_df, synth_df, real_meta_text, synth_meta_text


def sample_rows(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Sample N_ROWS rows reproducibly."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(N_ROWS, len(df)), replace=False)
    return df.iloc[sorted(idx)].reset_index(drop=True)


# ── Session state init ────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "step":         "setup",     # setup | trial_c1 | trial_c2 | pair_verdict | done
        "annotator_id": "",
        "session_id":   datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "dataset":      "adult",
        "method":       "ctgan",
        "n_pairs":      10,          # number of (C1,C2) sample pairs
        "queue":        [],          # list of (true_label, seed) tuples
        "pair_idx":     0,
        "phase":        "C1",        # C1 or C2 within a pair
        "rows_df":      None,        # current 20-row sample (shared C1/C2)
        "meta_text":    "",          # metadata text for current sample's source
        "true_label":   "",
        "c1_vote":      None,
        "c2_vote":      None,
        "c1_conf":      50,
        "c2_conf":      50,
        "c1_notes":     "",
        "c2_notes":     "",
        "records":      [],          # list of completed annotation dicts
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
S = st.session_state


# ── CSV save ──────────────────────────────────────────────────────────────────

def save_to_csv():
    if not S.records:
        return None
    df = pd.DataFrame(S.records)
    fname = RESULTS_DIR / f"human_results_{S.annotator_id}_{S.session_id}.csv"
    df.to_csv(fname, index=False)
    return fname, df


# ── Progress append ───────────────────────────────────────────────────────────

def commit_pair():
    """Save both C1 and C2 verdicts for the current pair."""
    true_lbl = S.true_label
    for cond, vote, conf, notes in [
        ("C1", S.c1_vote, S.c1_conf, S.c1_notes),
        ("C2", S.c2_vote, S.c2_conf, S.c2_notes),
    ]:
        S.records.append({
            "annotator_id":    S.annotator_id,
            "session_id":      S.session_id,
            "dataset":         S.dataset,
            "method":          S.method,
            "pair_id":         S.pair_idx,
            "condition":       cond,
            "n_rows_shown":    N_ROWS,
            "true_label":      true_lbl,
            "predicted_label": vote,
            "correct":         (vote == true_lbl),
            "confidence":      conf,
            "notes":           notes,
            "timestamp":       datetime.utcnow().isoformat(),
        })


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔬 Annotation Study")

    if S.step == "setup":
        st.markdown("**Study configuration**")
        S.annotator_id = st.text_input("Your annotator ID", placeholder="e.g. annotator_A")

        ds_options = {k: v["label"] for k, v in DATASETS.items()}
        ds_key = st.selectbox("Dataset", list(ds_options.keys()),
                              format_func=lambda x: ds_options[x])
        S.dataset = ds_key

        method_key = st.selectbox("Synthesis method", list(METHOD_LABELS.keys()),
                                  format_func=lambda x: METHOD_LABELS[x])
        S.method = method_key

        S.n_pairs = st.slider("Number of trial pairs (C1+C2 each)", 4, 20, 10)
        st.caption("Each pair shows the same 20-row sample under C1 (table only) then C2 (table + metadata).")

    else:
        st.markdown(f"**Annotator:** `{S.annotator_id}`")
        st.markdown(f"**Dataset:** {DATASETS[S.dataset]['label']}")
        st.markdown(f"**Method:** {METHOD_LABELS[S.method]}")
        st.markdown("---")

        if S.records:
            total   = len(S.records)
            correct = sum(r["correct"] for r in S.records)
            st.markdown(f"**Correct:** {correct}/{total} ({100*correct/total:.0f}%)")
            for cond in ("C1", "C2"):
                sub = [r for r in S.records if r["condition"] == cond]
                if sub:
                    c = sum(r["correct"] for r in sub)
                    st.markdown(f"  • {cond}: {c}/{len(sub)}")

        st.markdown(f"**Pair:** {S.pair_idx + 1} / {S.n_pairs}  |  Phase: **{S.phase}**")
        st.markdown("---")
        if st.button("↩ Restart", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ── SETUP SCREEN ──────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">Real or Synthetic?</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Human Annotation Study — Tabular Data Discrimination</div>',
            unsafe_allow_html=True)

if S.step == "setup":
    col1, col2, col3 = st.columns(3)
    for col, icon, text in [
        (col1, "🔍", "Inspect each 20-row table carefully"),
        (col2, "🧠", "Decide: REAL data or SYNTHETIC data?"),
        (col3, "📊", "Rate your confidence (0–100)"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#13161e;border:1px solid #2a2d38;border-radius:10px;
                        padding:1.2rem;text-align:center;">
              <div style="font-size:1.8rem">{icon}</div>
              <div style="color:#9ca3af;font-size:0.82rem;margin-top:0.4rem">{text}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
---
**How this works:**
- Each trial pair shows **the same 20-row sample** twice:
  1. **C1 — Table only**: judge from the raw data alone
  2. **C2 — Table + metadata**: judge with full distributional statistics added
- Your C1 verdict is locked before you see the metadata.
- This mirrors exactly how LLM models are tested in the study.

👈 Configure your session in the sidebar then click **Start** below.
""")

    if st.button("🚀  Start Annotation", use_container_width=True, type="primary"):
        if not S.annotator_id.strip():
            st.error("Please enter your annotator ID first.")
        else:
            # Build balanced shuffled queue of (label, seed) pairs
            rng = random.Random(abs(hash(S.annotator_id + S.dataset + S.method)) % (2**31))
            labels = ["REAL"] * S.n_pairs + ["SYNTHETIC"] * S.n_pairs
            rng.shuffle(labels)
            labels = labels[:S.n_pairs]          # take n_pairs of them (roughly balanced)
            seeds  = [rng.randint(0, 2**31) for _ in range(S.n_pairs)]
            S.queue  = list(zip(labels, seeds))
            S.pair_idx = 0
            S.phase  = "C1"
            S.c1_vote = None
            S.c2_vote = None
            S.step   = "trial"
            st.rerun()
    st.stop()


# ── TRIAL SCREEN ──────────────────────────────────────────────────────────────

if S.step == "done":
    st.success("🎉 Study complete! Thank you for your participation.")
    result = save_to_csv()
    if result:
        fname, df = result
        c = df["correct"].sum()
        n = len(df)
        st.markdown(f"### Final Score: **{c}/{n}** ({100*c/n:.1f}%)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**By condition:**")
            st.dataframe(
                df.groupby("condition")["correct"]
                  .agg(correct="sum", total="count")
                  .assign(accuracy=lambda x: (x["correct"]/x["total"]*100).round(1))
            )
        with col2:
            st.markdown("**By true label:**")
            st.dataframe(
                df.groupby("true_label")["correct"]
                  .agg(correct="sum", total="count")
                  .assign(accuracy=lambda x: (x["correct"]/x["total"]*100).round(1))
            )
        st.markdown(f"**Results saved to:** `{fname}`")
        st.download_button(
            "⬇  Download CSV",
            df.to_csv(index=False).encode(),
            file_name=fname.name,
            mime="text/csv",
            use_container_width=True,
        )
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    real_df, synth_df, real_meta, synth_meta = load_dataset(S.dataset, S.method)
except FileNotFoundError as e:
    st.error(f"Data file not found: {e}")
    st.stop()

# Ensure pair is loaded
if S.pair_idx >= S.n_pairs:
    S.step = "done"
    st.rerun()

true_label, seed = S.queue[S.pair_idx]
S.true_label = true_label

src_df  = real_df if true_label == "REAL" else synth_df
meta_text = real_meta if true_label == "REAL" else synth_meta

# Cache the 20-row sample for this pair (shared between C1 and C2)
sample_df = sample_rows(src_df, seed)

# ── Trial header ──────────────────────────────────────────────────────────────
progress = S.pair_idx / S.n_pairs
st.progress(progress)
st.markdown(
    f'<div class="trial-info">Pair {S.pair_idx+1} / {S.n_pairs} &nbsp;|&nbsp; '
    f'Dataset: {DATASETS[S.dataset]["label"]} &nbsp;|&nbsp; '
    f'Method: {METHOD_LABELS[S.method]}</div>',
    unsafe_allow_html=True,
)

phase = S.phase   # "C1" or "C2"

if phase == "C1":
    st.markdown('<span class="badge badge-c1">Condition C1 — Table only</span>',
                unsafe_allow_html=True)
    st.markdown("Examine the table below. **No distributional statistics are provided.**")
else:
    st.markdown('<span class="badge badge-c2">Condition C2 — Table + Metadata</span>',
                unsafe_allow_html=True)
    st.markdown("Same table as C1, now with full distributional metadata below.")
    if S.c1_vote:
        outcome = "✅ correct" if S.c1_vote == S.true_label else "❌ wrong"
        st.info(f"Your C1 verdict was **{S.c1_vote}** ({outcome}). "
                f"The metadata may help you reconsider.")

# ── Table display ─────────────────────────────────────────────────────────────
st.markdown(f"**Table ({N_ROWS} rows shown out of {len(src_df):,} total, "
            f"{len(src_df.columns)} columns)**")
st.dataframe(sample_df, use_container_width=True, height=420)

# C2: show metadata
if phase == "C2":
    with st.expander("📊 Distributional metadata (click to expand/collapse)", expanded=True):
        st.markdown(f'<div class="meta-block">{meta_text}</div>',
                    unsafe_allow_html=True)

# ── Vote form ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"### Your verdict for **C{1 if phase=='C1' else 2}**")

with st.form(key=f"vote_form_{S.pair_idx}_{phase}"):
    verdict = st.radio(
        "Is this table REAL or SYNTHETIC?",
        ["REAL", "SYNTHETIC"],
        horizontal=True,
        index=None,
    )
    conf = st.slider("Confidence (0 = complete guess, 100 = fully certain)", 0, 100, 50)
    notes = st.text_area(
        "Reasoning / notes (optional — what features guided your decision?)",
        placeholder="e.g. 'education and education_num mapping looks inconsistent'",
        height=80,
    )
    submitted = st.form_submit_button("Submit verdict →", use_container_width=True, type="primary")

if submitted:
    if verdict is None:
        st.warning("Please select REAL or SYNTHETIC before submitting.")
    else:
        if phase == "C1":
            S.c1_vote  = verdict
            S.c1_conf  = conf
            S.c1_notes = notes
            S.phase    = "C2"
        else:  # C2 — pair complete
            S.c2_vote  = verdict
            S.c2_conf  = conf
            S.c2_notes = notes
            commit_pair()
            # Advance to next pair
            S.pair_idx += 1
            S.phase     = "C1"
            S.c1_vote   = None
            S.c2_vote   = None
            if S.pair_idx >= S.n_pairs:
                S.step = "done"
        st.rerun()

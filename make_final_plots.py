#!/usr/bin/env python3
"""
make_final_plots.py
===================
Generates all publication-ready figures for both Adult and Census datasets.
Run from project root:
    python make_final_plots.py

Outputs (all in results/):
    fig_drs_by_method_condition.pdf          — Adult DRS (updated)
    fig_census_drs_by_method_condition.pdf   — Census DRS
    fig_dataset_comparison.pdf               — Adult vs Census side-by-side
    fig_model_accuracy_both_datasets.pdf     — Provider accuracy comparison
    fig_census_formal_privacy_correlation.pdf — Census DRS vs formal measures
"""

import json
import glob
import collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path("/Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026")
RESULTS_ADULT  = ROOT / "results" / "adult"
RESULTS_CENSUS = ROOT / "results" / "census"
OUT_DIR        = ROOT / "results"

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        15,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  12,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

METHOD_ORDER   = ["ctgan", "tvae", "gaussian_copula"]
METHOD_LABELS  = {"ctgan": "CTGAN", "tvae": "TVAE", "gaussian_copula": "Gaussian\nCopula"}
C1_COLOR       = "#2166ac"
C2_COLOR       = "#d6604d"
GOOGLE_COLOR   = "#4285F4"
GROQ_COLOR     = "#FF6B35"
ADULT_COLOR    = "#1b7837"
CENSUS_COLOR   = "#762a83"

VALID_LABELS = {"REAL", "SYNTHETIC"}
APPROVED_MODELS = {
    ("groq", "llama-3.1-8b-instant"): "Groq",
    ("google", "gemini-2.5-flash"): "Google",
}
EXCLUDED_PATH_MARKERS = ("archive_invalid", "/res1/")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir, filter_errors=True):
    rows = []
    for f in sorted(glob.glob(str(Path(results_dir) / "**/*.jsonl"), recursive=True)):
        f_norm = f.replace("\\", "/")
        if any(marker in f_norm for marker in EXCLUDED_PATH_MARKERS):
            continue

        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)

                    pred = r.get("predicted_label")
                    true = r.get("true_label")
                    if filter_errors and pred == "ERROR":
                        continue
                    if pred not in VALID_LABELS or true not in VALID_LABELS:
                        continue

                    model_key = (r.get("provider"), r.get("model"))
                    if model_key not in APPROVED_MODELS:
                        continue

                    r["provider_label"] = APPROVED_MODELS[model_key]
                    rows.append(r)
                except Exception:
                    pass
    return rows


def compute_drs(rows, provider=None):
    """
    Compute DRS = min(p_real, p_synthetic) per (method, condition) cell.
    Optionally filter to a single provider.
    Returns dict keyed by (method, condition) → {drs, p_real, p_syn, n_real, n_syn}.
    """
    if provider:
        rows = [
            r
            for r in rows
            if r.get("provider_label", r.get("provider")) == provider
        ]

    cells = collections.defaultdict(lambda: {"nr": 0, "cr": 0, "ns": 0, "cs": 0})
    for r in rows:
        method = r.get("synthetic_method", r.get("method", "unknown"))
        cond   = r.get("condition", "?")
        tl     = r.get("true_label", "?")
        ok     = int(r.get("correct", False))
        key    = (method, cond)
        if tl == "REAL":
            cells[key]["nr"] += 1
            cells[key]["cr"] += ok
        else:
            cells[key]["ns"] += 1
            cells[key]["cs"] += ok

    out = {}
    for key, d in cells.items():
        p_real = d["cr"] / d["nr"] if d["nr"] else 0.0
        p_syn  = d["cs"] / d["ns"] if d["ns"] else 0.0
        out[key] = {
            "drs":   min(p_real, p_syn),
            "p_real": p_real,
            "p_syn":  p_syn,
            "n_real": d["nr"],
            "n_syn":  d["ns"],
            "n":      d["nr"] + d["ns"],
        }
    return out


def provider_summary(rows):
    """Accuracy by provider."""
    by_prov = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        prov = r.get("provider_label", r.get("provider", "?"))
        by_prov[prov]["n"] += 1
        by_prov[prov]["correct"] += int(r.get("correct", False))
    return {p: d["correct"] / d["n"] for p, d in by_prov.items() if d["n"] > 0}


# ── Figure helpers ────────────────────────────────────────────────────────────

def drs_bar_chart(ax, drs_dict, title, methods=None, show_n=True):
    """Draw a grouped bar chart (C1 / C2) for a single dataset."""
    if methods is None:
        methods = METHOD_ORDER
    x    = np.arange(len(methods))
    w    = 0.38
    bars_c1, bars_c2 = [], []

    for i, m in enumerate(methods):
        v_c1  = drs_dict.get((m, "C1"), {}).get("drs", 0.0) * 100
        v_c2  = drs_dict.get((m, "C2"), {}).get("drs", 0.0) * 100
        n_c1  = drs_dict.get((m, "C1"), {}).get("n", 0)
        n_c2  = drs_dict.get((m, "C2"), {}).get("n", 0)

        b1 = ax.bar(i - w / 2, v_c1, width=w, color=C1_COLOR, alpha=0.85,
                    edgecolor="white", linewidth=0.5, zorder=3)
        b2 = ax.bar(i + w / 2, v_c2, width=w, color=C2_COLOR, alpha=0.85,
                    edgecolor="white", linewidth=0.5, zorder=3)
        bars_c1.append(b1)
        bars_c2.append(b2)

        # N labels
        if show_n and n_c1:
            ax.text(i - w / 2, v_c1 + 0.8, f"N={n_c1}", ha="center",
                    va="bottom", fontsize=14, fontweight='bold', color="dimgray")
        if show_n and n_c2:
            ax.text(i + w / 2, v_c2 + 0.8, f"N={n_c2}", ha="center",
                    va="bottom", fontsize=14, color="dimgray")

    ax.axhline(50, color="gray", lw=1, ls="--", alpha=0.5, zorder=2, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("Disclosure Risk Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color=C1_COLOR, alpha=0.85, label="C1 (table only)"),
        mpatches.Patch(color=C2_COLOR, alpha=0.85, label="C2 (table + metadata)"),
        plt.Line2D([0], [0], color="gray", ls="--", lw=1, label="Chance (50%)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)


# ── Figure 1: Adult DRS (regenerated for consistency) ────────────────────────

def plot_adult_drs(adult_rows):
    drs = compute_drs(adult_rows)   # all providers combined
    fig, ax = plt.subplots(figsize=(6.5, 4))
    drs_bar_chart(ax, drs, f"Adult Dataset — Disclosure Risk Score by Method & Condition\n(N={len(adult_rows)} valid trials, approved providers)")
    fig.tight_layout()
    out = OUT_DIR / "fig_adult_drs_by_method_condition.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"[OK] {out.name}")
    plt.close(fig)
    return drs


# ── Figure 2: Census DRS ──────────────────────────────────────────────────────

def plot_census_drs(census_rows):
    drs = compute_drs(census_rows)  # all providers combined
    fig, ax = plt.subplots(figsize=(6.5, 4))
    drs_bar_chart(ax, drs, f"Census Dataset — Disclosure Risk Score by Method & Condition\n(N={len(census_rows)} valid trials, approved providers)")
    fig.tight_layout()
    out = OUT_DIR / "fig_census_drs_by_method_condition.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"[OK] {out.name}")
    plt.close(fig)
    return drs


# ── Figure 3: Adult vs Census side-by-side ────────────────────────────────────

def plot_dataset_comparison(adult_rows, census_rows):
    drs_adult  = compute_drs(adult_rows)
    drs_census = compute_drs(census_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    drs_bar_chart(axes[0], drs_adult,
                  f"Adult")
    drs_bar_chart(axes[1], drs_census,
                  f"Census")
    axes[1].set_ylabel("")
    # fig.suptitle("Disclosure Risk Score — Adult vs Census Comparison", fontsize=12, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig_dataset_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"[OK] {out.name}")
    plt.close(fig)


# ── Figure 4: Provider accuracy comparison (both datasets) ────────────────────

def plot_provider_accuracy(adult_rows, census_rows):
    def acc_by_provider(rows):
        d = collections.defaultdict(lambda: {"n": 0, "ok": 0})
        for r in rows:
            p = r.get("provider_label", r.get("provider", "?"))
            d[p]["n"]  += 1
            d[p]["ok"] += int(r.get("correct", False))
        return {p: v["ok"] / v["n"] * 100 for p, v in d.items() if v["n"] >= 5}

    acc_adult  = acc_by_provider(adult_rows)
    acc_census = acc_by_provider(census_rows)

    providers = sorted(set(acc_adult) | set(acc_census))
    x = np.arange(len(providers))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, prov in enumerate(providers):
        va = acc_adult.get(prov, None)
        vc = acc_census.get(prov, None)
        label_a = f"{va:.1f}%" if va is not None else "—"
        label_c = f"{vc:.1f}%" if vc is not None else "—"
        n_a = sum(1 for r in adult_rows  if r.get("provider_label", r.get("provider")) == prov)
        n_c = sum(1 for r in census_rows if r.get("provider_label", r.get("provider")) == prov)

        if va is not None:
            b = ax.bar(i - w / 2, va, width=w, color=ADULT_COLOR,  alpha=0.85,
                       edgecolor="white", zorder=3)
            ax.text(i - w / 2, va + 0.5, f"{va:.0f}%\n(N={n_a})",
                    ha="center", va="bottom", fontsize=14, fontweight='bold', color="dimgray")
        if vc is not None:
            b = ax.bar(i + w / 2, vc, width=w, color=CENSUS_COLOR, alpha=0.85,
                       edgecolor="white", zorder=3)
            ax.text(i + w / 2, vc + 0.5, f"{vc:.0f}%\n(N={n_c})",
                    ha="center", va="bottom", fontsize=14, fontweight='bold', color="dimgray")

    ax.axhline(50, color="gray", lw=1, ls="--", alpha=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(providers)
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("LLM Discrimination Accuracy by Provider\nAdult vs Census")
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color=ADULT_COLOR,  alpha=0.85, label="Adult dataset"),
        mpatches.Patch(color=CENSUS_COLOR, alpha=0.85, label="Census dataset"),
        plt.Line2D([0], [0], color="gray", ls="--", lw=1, label="Chance (50%)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out = OUT_DIR / "fig_model_accuracy_both_datasets.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"[OK] {out.name}")
    plt.close(fig)


# ── Figure 5: Census formal privacy correlation ───────────────────────────────

def plot_census_correlation(census_rows):
    try:
        df_corr = pd.read_csv(RESULTS_CENSUS / "formal_privacy_correlation.csv")
    except Exception as e:
        print(f"[SKIP] Census correlation CSV not found: {e}")
        return

    # Compute per-method DRS average from LLM trials
    drs = compute_drs(census_rows)
    method_drs = {}
    for m in METHOD_ORDER:
        vals = [drs.get((m, c), {}).get("drs", np.nan) for c in ["C1", "C2"]]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            method_drs[m] = np.mean(vals) * 100

    # Canonical method names in the CSV vs in code
    name_map = {"CTGAN": "ctgan", "GaussianCopula": "gaussian_copula", "TVAE": "tvae"}
    methods_csv = df_corr["Method"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, (xcol, xlabel) in zip(axes, [
        ("MIA_Discrimination", "MIA Score"),
        ("Linkage_Success_Rate", "Linkage Risk"),
    ]):
        xs, ys, labels = [], [], []
        for _, row in df_corr.iterrows():
            mcode = name_map.get(row["Method"], row["Method"].lower())
            if mcode in method_drs:
                xs.append(row[xcol])
                ys.append(method_drs[mcode])
                labels.append(row["Method"])

        ax.scatter(xs, ys, color=CENSUS_COLOR, s=80, zorder=3)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), xytext=(5, 3), textcoords="offset points",
                        fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("LLM DRS Avg (%)")
        ax.set_title(f"Census: DRS vs {xlabel}")
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

    # fig.suptitle("Census Dataset — LLM DRS vs Formal Privacy Measures\n(N=3 methods; directional evidence only)",
                #  fontsize=10, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig_census_formal_privacy_correlation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"[OK] {out.name}")
    plt.close(fig)


# ── Summary printout ──────────────────────────────────────────────────────────

def print_summary(label, rows, drs_all):
    print(f"\n{'='*60}")
    print(f"  {label}  ({len(rows)} valid trials)")
    print(f"{'='*60}")
    by_prov = collections.defaultdict(lambda: {"n": 0, "ok": 0})
    for r in rows:
        p = r.get("provider_label", r.get("provider", "?"))
        by_prov[p]["n"]  += 1
        by_prov[p]["ok"] += int(r.get("correct", False))
    for p, d in sorted(by_prov.items()):
        print(f"  {p:12s}: {d['ok']:3d}/{d['n']:3d}  ({d['ok']/d['n']:.1%})")

    print(f"\n  DRS by Method × Condition:")
    print(f"  {'Method':<18} {'C1 DRS':>8} {'C2 DRS':>8}  "
          f"{'N_C1':>5} {'N_C2':>5}  {'p_real_C1':>10} {'p_syn_C1':>9}")
    print(f"  {'-'*18} {'-'*8} {'-'*8}  {'-'*5} {'-'*5}  {'-'*10} {'-'*9}")
    for m in METHOD_ORDER:
        c1 = drs_all.get((m, "C1"), {})
        c2 = drs_all.get((m, "C2"), {})
        print(f"  {METHOD_LABELS[m].replace(chr(10),' '):<18} "
              f"{c1.get('drs',0)*100:>7.1f}% "
              f"{c2.get('drs',0)*100:>7.1f}%  "
              f"{c1.get('n',0):>5} {c2.get('n',0):>5}  "
              f"{c1.get('p_real',0)*100:>9.1f}% "
              f"{c1.get('p_syn',0)*100:>8.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    adult_rows  = load_results(RESULTS_ADULT)
    census_rows = load_results(RESULTS_CENSUS)
    print(f"  Adult:  {len(adult_rows)} valid trials")
    print(f"  Census: {len(census_rows)} valid trials")

    print("\nGenerating plots...")
    drs_adult  = plot_adult_drs(adult_rows)
    drs_census = plot_census_drs(census_rows)
    plot_dataset_comparison(adult_rows, census_rows)
    plot_provider_accuracy(adult_rows, census_rows)
    plot_census_correlation(census_rows)

    print_summary("ADULT",  adult_rows,  drs_adult)
    print_summary("CENSUS", census_rows, drs_census)

    # Print census DRS values formatted for paper
    print(f"\n{'='*60}")
    print("  CENSUS DRS VALUES FOR PAPER")
    print(f"{'='*60}")
    for m in METHOD_ORDER:
        c1 = drs_census.get((m, "C1"), {})
        c2 = drs_census.get((m, "C2"), {})
        avg = (c1.get("drs", 0) + c2.get("drs", 0)) / 2
        print(f"  {METHOD_LABELS[m].replace(chr(10),' '):<18}: "
              f"C1={c1.get('drs',0)*100:.1f}%  C2={c2.get('drs',0)*100:.1f}%  "
              f"Avg={avg*100:.1f}%  "
              f"(N_C1={c1.get('n',0)}, N_C2={c2.get('n',0)})")

    print(f"\n  Done. All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

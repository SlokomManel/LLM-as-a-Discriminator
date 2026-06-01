"""
analyze_reasoning.py
====================
Extracts and categorises the qualitative reasoning emitted by LLMs
in the tabular discrimination experiment.

Analyses red_flags, reasoning, and supporting_evidence fields from
all adult JSONL result files and prints a structured summary with
counts, representative quotes, and per-provider breakdowns.

Run from project root:
    python analyze_reasoning.py
"""

import json, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent

# ── Load records ──────────────────────────────────────────────────────────────

def load_adult_records():
    """Load all valid verdict records from Adult experiment JSONL files."""
    adult_dir = ROOT / "results" / "adult"
    records = []
    for jf in sorted(adult_dir.glob("llm_results_*.jsonl")):
        stem = jf.stem  # e.g. llm_results_ctgan_google_20260526_164854
        provider = "google" if "google" in stem else "groq"
        with open(jf) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("error") or not obj.get("predicted_label"):
                    continue
                records.append({
                    "provider":    provider,
                    "model":       obj.get("model", ""),
                    "method":      obj.get("synthetic_method", ""),
                    "condition":   obj.get("condition", ""),
                    "true_label":  obj.get("true_label", ""),
                    "verdict":     obj.get("predicted_label", ""),
                    "correct":     obj.get("correct", False),
                    "confidence":  obj.get("confidence", 50),
                    "red_flags":   obj.get("red_flags", []),
                    "reasoning":   obj.get("reasoning", ""),
                    "supporting":  obj.get("supporting_evidence", []),
                })
    return records


# ── Thematic categorisation ───────────────────────────────────────────────────

# Each theme → list of keyword patterns to search in reasoning + red_flags text
THEMES = {
    "Education–education_num mapping": [
        "education_num", "education-num", "education num",
        "functional dependency", "bijective", "1:1 mapping",
        "mapping between", "10th", "bachelors", "hs-grad",
    ],
    "Marital/relationship contradiction": [
        "marital", "relationship", "never-married", "husband",
        "wife", "unmarried",
    ],
    "Capital gain/loss anomaly": [
        "capital_gain", "capital_loss", "capital gain", "capital loss",
        "small non-zero", "small integer",
    ],
    "Distribution irregularities": [
        "distribution", "skewness", "skew", "kurtosis",
        "non-normal", "uniform", "shapiro",
    ],
    "Dataset fingerprinting (UCI Adult)": [
        "30,162", "30162", "99,999", "99999",
        "UCI", "Adult dataset", "census income",
        "known dataset", "original dataset",
    ],
    "Zero / sparse values": [
        "large number of zeros", "zero values", "many zeros",
        "sparse", "zero in capital",
    ],
    "Age distribution": [
        "age distribution", "age range", "unrealistic age",
        "young age", "old age",
    ],
    "Syntactic / structural artifact": [
        "implausible", "inconsistent", "suspicious", "anomal",
        "artifact", "artefact", "regularit",
    ],
}


def match_theme(record: dict, keywords: list[str]) -> bool:
    full_text = " ".join(record["red_flags"]) + " " + record["reasoning"]
    full_text = full_text.lower()
    return any(kw.lower() in full_text for kw in keywords)


# ── Summary printing ──────────────────────────────────────────────────────────

def print_section(title: str, width: int = 80):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print('=' * width)


def analyze(records: list[dict]):
    n = len(records)
    print(f"\nTotal valid Adult records loaded: {n}")

    # ── Provider counts ────────────────────────────────────────────────────────
    print_section("Provider breakdown")
    for prov in ("google", "groq"):
        sub = [r for r in records if r["provider"] == prov]
        correct = sum(r["correct"] for r in sub)
        print(f"  {prov:8s}: {len(sub):4d} records, accuracy {100*correct/len(sub):.1f}%")

    # ── Thematic analysis ─────────────────────────────────────────────────────
    print_section("Thematic category counts (# records mentioning theme)")
    theme_counts = {}
    for theme, keywords in THEMES.items():
        hits = sum(1 for r in records if match_theme(r, keywords))
        theme_counts[theme] = hits

    for theme, cnt in sorted(theme_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / n
        print(f"  {pct:5.1f}%  ({cnt:3d}/{n})  {theme}")

    # ── Per-provider theme breakdown ──────────────────────────────────────────
    print_section("Themes by provider (% of that provider's records)")
    for prov in ("google", "groq"):
        sub = [r for r in records if r["provider"] == prov]
        print(f"\n  [{prov}]  n={len(sub)}")
        for theme, keywords in THEMES.items():
            hits = sum(1 for r in sub if match_theme(r, keywords))
            print(f"    {100*hits/len(sub):5.1f}%  {theme}")

    # ── Top raw red_flag phrases ───────────────────────────────────────────────
    print_section("Top 20 most frequent raw red_flag strings")
    raw_flags: Counter = Counter()
    for r in records:
        for f in r["red_flags"]:
            raw_flags[f.lower()[:90]] += 1
    for phrase, cnt in raw_flags.most_common(20):
        print(f"  {cnt:3d}x  {phrase}")

    # ── Representative quotes per theme ───────────────────────────────────────
    print_section("Representative reasoning extracts per theme (1 per theme)")
    for theme, keywords in THEMES.items():
        hits = [r for r in records if match_theme(r, keywords)]
        if not hits:
            print(f"\n  [{theme}]  — no examples found")
            continue
        # Prefer a Gemini record (richer reasoning); prefer correct ones
        hits.sort(key=lambda r: (r["provider"] != "google", not r["correct"]))
        example = hits[0]
        snippet = example["reasoning"][:400].replace("\n", " ")
        flags_str = "; ".join(example["red_flags"][:3])
        print(f"\n  [{theme}]  provider={example['provider']}, "
              f"method={example['method']}, cond={example['condition']}, "
              f"correct={example['correct']}")
        print(f"    red_flags: {flags_str[:120]}")
        print(f"    reasoning: {snippet}…")

    # ── Accuracy by verdict tendency ──────────────────────────────────────────
    print_section("Verdict breakdown (correct detection vs. bias)")
    for prov in ("google", "groq"):
        sub = [r for r in records if r["provider"] == prov]
        real_recs = [r for r in sub if r["true_label"] == "REAL"]
        syn_recs  = [r for r in sub if r["true_label"] == "SYNTHETIC"]
        real_acc  = 100 * sum(r["correct"] for r in real_recs)  / max(len(real_recs), 1)
        syn_acc   = 100 * sum(r["correct"] for r in syn_recs)   / max(len(syn_recs),  1)
        print(f"  {prov:8s}:  REAL-correct={real_acc:.1f}%  SYNTHETIC-correct={syn_acc:.1f}%")

    # ── Condition effect on themes ─────────────────────────────────────────────
    print_section("Education–education_num theme appearance: C1 vs C2 (Gemini only)")
    edu_kws = THEMES["Education–education_num mapping"]
    google_recs = [r for r in records if r["provider"] == "google"]
    for cond in ("C1", "C2"):
        sub = [r for r in google_recs if r["condition"] == cond]
        hits = sum(1 for r in sub if match_theme(r, edu_kws))
        if sub:
            print(f"  {cond}: {hits}/{len(sub)} ({100*hits/len(sub):.1f}%) mention edu mapping")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    records = load_adult_records()
    if not records:
        print("No records found. Check that results/adult/*.jsonl files exist.", file=sys.stderr)
        sys.exit(1)
    analyze(records)

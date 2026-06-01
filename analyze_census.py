# /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026/analyze_census.py
import json, glob, collections, os
from pathlib import Path

ROOT = Path("/Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026")

def load_results(results_dir):
    rows = []
    for f in sorted(glob.glob(str(results_dir / "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("predicted_label", "ERROR") != "ERROR":
                    rows.append(r)
            except:
                pass
    return rows

def summarize(rows, label):
    if not rows:
        print(f"\n{label}: No valid results yet.")
        return
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    by_method_cond = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    by_method = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    by_cond = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    by_provider = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    
    for r in rows:
        method = r.get("synthetic_method", r.get("method", "unknown"))
        cond = r.get("condition", "?")
        prov = r.get("provider", "?")
        correct = int(r.get("correct", False))
        
        key = (method, cond)
        by_method_cond[key]["n"] += 1
        by_method_cond[key]["correct"] += correct
        by_method[method]["n"] += 1
        by_method[method]["correct"] += correct
        by_cond[cond]["n"] += 1
        by_cond[cond]["correct"] += correct
        by_provider[prov]["n"] += 1
        by_provider[prov]["correct"] += correct
    
    total = len(rows)
    overall_acc = sum(r.get("correct", False) for r in rows) / total
    print(f"  Total valid trials: {total}")
    print(f"  Overall accuracy:   {overall_acc:.1%}")
    
    print(f"\n  By Provider:")
    for prov, d in sorted(by_provider.items()):
        acc = d["correct"] / d["n"]
        print(f"    {prov:20s}  {d['correct']:3d}/{d['n']:3d}  ({acc:.1%})")
    
    print(f"\n  By Method:")
    for method, d in sorted(by_method.items()):
        acc = d["correct"] / d["n"]
        print(f"    {method:<20s}  {d['correct']:3d}/{d['n']:3d}  ({acc:.1%})")
    
    print(f"\n  By Condition:")
    for cond, d in sorted(by_cond.items()):
        acc = d["correct"] / d["n"]
        print(f"    {cond:20s}  {d['correct']:3d}/{d['n']:3d}  ({acc:.1%})")
    
    print(f"\n  By Method x Condition:")
    print(f"    {'Method':<20s}  {'Cond':4s}  {'n':>4s}  {'Acc':>6s}")
    print(f"    {'-'*20}  {'-'*4}  {'-'*4}  {'-'*6}")
    for (method, cond), d in sorted(by_method_cond.items()):
        acc = d["correct"] / d["n"]
        print(f"    {method:<20s}  {cond:4s}  {d['n']:>4d}  {acc:>6.1%}")

def show_census_progress():
    census_dir = ROOT / "results" / "census"
    print(f"\n{'='*60}")
    print(f"  CENSUS EXPERIMENT PROGRESS")
    print(f"{'='*60}")
    
    methods = ["ctgan", "tvae", "gaussian_copula"]
    providers = ["google", "groq"]
    
    for method in methods:
        for prov in providers:
            files = sorted(glob.glob(str(census_dir / f"llm_results_{method}_{prov}_*.jsonl")))
            valid = 0
            errors = 0
            for f in files:
                for line in open(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if r.get("predicted_label", "ERROR") == "ERROR":
                            errors += 1
                        else:
                            valid += 1
                    except:
                        pass
            status = "DONE" if (valid >= 20 and errors == 0) else ("PARTIAL" if valid > 0 else ("QUOTA HIT" if errors > 0 else "PENDING"))
            print(f"  {method:20s} | {prov:8s} | {status:12s} | valid={valid:3d}  errors={errors:3d}")

# Run analysis
print("\n" + "="*60)
print("  RESULTS ANALYSIS — Census vs Adult")
print("="*60)

show_census_progress()

census_rows = load_results(ROOT / "results" / "census")
adult_rows = load_results(ROOT / "results" / "adult")

summarize(census_rows, "CENSUS RESULTS (so far — Google gemini-2.5-flash)")
summarize(adult_rows, "ADULT RESULTS (complete — Google + Groq)")

print()

"""
evaluate.py
Runs LLM discrimination trials across both conditions and both providers.
Saves structured results to results/llm_results.jsonl
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Literal

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from prompt_builder import build_prompt_c1, build_prompt_c2

load_dotenv()

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Utilities ──────────────────────────────────────────────────────────────────

def df_to_markdown(df: pd.DataFrame, n_rows: int = 20) -> str:
    """Convert DataFrame to markdown table (sample if needed)."""
    if len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=42).reset_index(drop=True)
    return df.to_markdown(index=False)


# ── LLM Callers ───────────────────────────────────────────────────────────────

def call_openai(system: str, user: str, model: str = "gpt-4o") -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def call_anthropic(system: str, user: str, model: str = "claude-opus-4-5-20251101") -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system + "\n\nIMPORTANT: Respond ONLY with valid JSON.",
        messages=[{"role": "user", "content": user}],
        temperature=0,
    )
    raw = message.content[0].text
    # Strip markdown fences if present
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(raw)


def call_github_models(system: str, user: str, model: str = "gpt-4o") -> dict:
    """Use GitHub Models via OpenAI-compatible API."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GITHUB_PAT"),
        base_url="https://models.inference.ai.azure.com"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def call_google(system: str, user: str, model: str = "gemini-2.5-flash") -> dict:
    """Use Google Gemini via Google AI Studio."""
    import google.generativeai as genai
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Map shorthand names to available Gemini models
    model_map = {
        "gemini-1.5-flash": "gemini-2.5-flash",  # Use 2.5 instead of deprecated 1.5
        "gemini-1.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "flash": "gemini-2.5-flash",
        "pro": "gemini-2.5-pro",
    }
    
    full_model_name = model_map.get(model, model)
    
    try:
        model_obj = genai.GenerativeModel(
            model_name=full_model_name,
            system_instruction=system
        )
        response = model_obj.generate_content(
            user + "\n\nIMPORTANT: Respond ONLY with valid JSON.",
            generation_config=genai.types.GenerationConfig(temperature=0)
        )
        raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Google API error with {full_model_name}: {e}. Retrying with gemini-2.5-pro...")
        # Fallback to gemini-2.5-pro if flash fails
        try:
            model_obj = genai.GenerativeModel(
                model_name="gemini-2.5-pro",
                system_instruction=system
            )
            response = model_obj.generate_content(
                user + "\n\nIMPORTANT: Respond ONLY with valid JSON.",
                generation_config=genai.types.GenerationConfig(temperature=0)
            )
            raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception as e2:
            raise Exception(f"Google API failed (both models): {e2}")


def call_groq(system: str, user: str, model: str = "llama-3.1-70b-versatile") -> dict:
    """Use Groq Cloud API."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


PROVIDER_MAP = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "github": call_github_models,
    "google": call_google,
    "groq": call_groq,
}


# ── Trial Runner ──────────────────────────────────────────────────────────────

def run_trial(
    df: pd.DataFrame,
    true_label: Literal["REAL", "SYNTHETIC"],
    condition: Literal["C1", "C2"],
    provider: str,
    model: str,
    trial_id: int,
    n_rows_shown: int = 20,
    retry_on_fail: bool = True,
) -> dict:
    """Run a single discrimination trial with shuffled dataset labels."""
    
    # Shuffle: randomly decide if this df is "Dataset A" or "Dataset B"
    # and whether the LLM should guess which is REAL
    use_shuffle = random.choice([True, False])
    
    if condition == "C1":
        system, user = build_prompt_c1(df, n_rows_shown=n_rows_shown)
    else:
        system, user = build_prompt_c2(df, n_rows_shown=n_rows_shown, label="unknown")

    call_fn = PROVIDER_MAP[provider]

    try:
        response = call_fn(system, user, model=model)
    except Exception as e:
        logger.warning(f"Trial {trial_id} failed: {e}. Retrying once...")
        if retry_on_fail:
            time.sleep(2)
            try:
                response = call_fn(system, user, model=model)
            except Exception as e2:
                logger.error(f"Trial {trial_id} failed twice: {e2}")
                response = {
                    "verdict": "ERROR",
                    "confidence": -1,
                    "reasoning": str(e2),
                    "red_flags": [],
                    "supporting_evidence": [],
                }
        else:
            response = {
                "verdict": "ERROR",
                "confidence": -1,
                "reasoning": str(e),
                "red_flags": [],
                "supporting_evidence": [],
            }

    verdict = response.get("verdict", "ERROR").upper()
    correct = verdict == true_label

    result = {
        "trial_id": trial_id,
        "timestamp": datetime.utcnow().isoformat(),
        "provider": provider,
        "model": model,
        "condition": condition,
        "true_label": true_label,
        "predicted_label": verdict,
        "correct": correct,
        "confidence": response.get("confidence", -1),
        "reasoning": response.get("reasoning", ""),
        "red_flags": response.get("red_flags", []),
        "supporting_evidence": response.get("supporting_evidence", []),
    }

    return result


# ── Main Experiment Loop ──────────────────────────────────────────────────────

def run_experiment_single_method(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    method_name: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    n_trials: int = 10,
    conditions: list = ["C1", "C2"],
    n_rows_shown: int = 20,
    sample_rows: int = None,
    output_file: str = None,
):
    """
    Run discrimination trials for a specific synthetic method.
    """
    if output_file is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"llm_results_{method_name}_{provider}_{ts}.jsonl"

    all_results = []
    trial_id = 0

    # Build trial queue: balanced real/synthetic per condition
    trial_queue = []
    for condition in conditions:
        for _ in range(n_trials // 2):
            trial_queue.append(("REAL", condition))
            trial_queue.append(("SYNTHETIC", condition))

    random.shuffle(trial_queue)

    logger.info(f"Running {len(trial_queue)} trials for {method_name}...")
    with open(output_file, "w") as f:
        for true_label, condition in tqdm(trial_queue, desc=f"{method_name} | {provider} / {model}"):
            df = real_df if true_label == "REAL" else synthetic_df

            # Optionally sample a subset of rows for variety
            if sample_rows and sample_rows < len(df):
                df = df.sample(n=sample_rows, random_state=trial_id).reset_index(drop=True)

            result = run_trial(
                df=df,
                true_label=true_label,
                condition=condition,
                provider=provider,
                model=model,
                trial_id=trial_id,
                n_rows_shown=n_rows_shown,
            )
            
            # Add method name to result
            result["synthetic_method"] = method_name

            f.write(json.dumps(result) + "\n")
            f.flush()
            all_results.append(result)

            logger.info(
                f"Trial {trial_id} | {method_name} | {condition} | label={true_label} | "
                f"pred={result['predicted_label']} | correct={result['correct']} | "
                f"conf={result['confidence']}"
            )
            trial_id += 1
            time.sleep(0.5)  # polite rate limiting

    # Summary stats
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby(["condition", "true_label"])["correct"].agg(["sum", "count", "mean"])
    logger.info(f"\n=== SUMMARY ({method_name}) ===\n{summary}")

    summary_path = str(output_file).replace(".jsonl", "_summary.csv")
    results_df.to_csv(summary_path, index=False)
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Summary saved to {summary_path}")

    return all_results


def run_experiment(
    real_path: str,
    synthetic_dir: str = None,
    synthetic_path: str = None,
    synthetic_methods: list = None,
    provider: str = "openai",
    model: str = "gpt-4o",
    n_trials: int = 10,
    conditions: list = ["C1", "C2"],
    n_rows_shown: int = 20,
    sample_rows: int = None,
):
    """
    Run the full discrimination experiment across multiple synthetic methods.

    Can be called in two ways:
    1. Multi-method: Pass synthetic_dir and optionally synthetic_methods list
    2. Single method (legacy): Pass synthetic_path
    """
    real_df = pd.read_csv(real_path)
    logger.info(f"Real data loaded: {real_df.shape}")
    
    # Determine which methods to run
    methods_to_run = []
    
    if synthetic_path:
        # Legacy single-method mode
        logger.info("Single-method mode (legacy)")
        methods_to_run = [("single", synthetic_path)]
    else:
        # Multi-method mode
        if synthetic_dir is None:
            synthetic_dir = "data/synthetic"
        
        synthetic_dir = Path(synthetic_dir)
        if not synthetic_dir.exists():
            logger.error(f"Synthetic directory not found: {synthetic_dir}")
            return
        
        # Get list of methods to run
        if synthetic_methods is None:
            # Auto-discover all methods with synthetic.csv files
            synthetic_methods = sorted([
                d.name for d in synthetic_dir.iterdir()
                if d.is_dir() and (d / "synthetic.csv").exists()
            ])
        
        if not synthetic_methods:
            logger.warning(f"No synthetic data files found in {synthetic_dir}")
            logger.warning("Expected structure: {synthetic_dir}/{method_name}/synthetic.csv")
            return
        
        logger.info(f"Found {len(synthetic_methods)} synthetic method(s): {synthetic_methods}")
        methods_to_run = [(m, synthetic_dir / m / "synthetic.csv") for m in synthetic_methods]
    
    # Run trials for each method
    all_aggregated_results = []
    for method_name, method_path in methods_to_run:
        try:
            synthetic_df = pd.read_csv(method_path)
            logger.info(f"\nLoaded {method_name}: {synthetic_df.shape}")
            
            results = run_experiment_single_method(
                real_df=real_df,
                synthetic_df=synthetic_df,
                method_name=method_name,
                provider=provider,
                model=model,
                n_trials=n_trials,
                conditions=conditions,
                n_rows_shown=n_rows_shown,
                sample_rows=sample_rows,
            )
            all_aggregated_results.extend(results)
        except Exception as e:
            logger.error(f"Failed to run experiment for {method_name}: {e}")
            continue
    
    # Aggregate summary across all methods
    if all_aggregated_results:
        logger.info(f"\n\n{'='*60}")
        logger.info(f"AGGREGATE SUMMARY (All Methods)")
        logger.info(f"{'='*60}")
        aggregated_df = pd.DataFrame(all_aggregated_results)
        
        # Overall accuracy
        overall_accuracy = aggregated_df["correct"].mean()
        logger.info(f"Overall Accuracy: {overall_accuracy:.3f}")
        
        # By method
        logger.info(f"\nAccuracy by Method:")
        method_acc = aggregated_df.groupby("synthetic_method")["correct"].agg(["sum", "count", "mean"])
        logger.info(f"\n{method_acc}")
        
        # By condition
        logger.info(f"\nAccuracy by Condition:")
        cond_acc = aggregated_df.groupby("condition")["correct"].agg(["sum", "count", "mean"])
        logger.info(f"\n{cond_acc}")
        
        # By method and condition
        logger.info(f"\nAccuracy by Method and Condition:")
        method_cond_acc = aggregated_df.groupby(["synthetic_method", "condition"])["correct"].agg(["sum", "count", "mean"])
        logger.info(f"\n{method_cond_acc}")
    
    return all_aggregated_results


def run_experiment_legacy(
    real_path: str,
    synthetic_path: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    n_trials: int = 10,
    conditions: list = ["C1", "C2"],
    n_rows_shown: int = 20,
    sample_rows: int = None,
    output_file: str = None,
):
    """
    Legacy single-method experiment (backwards compatible).
    """
    real_df = pd.read_csv(real_path)
    synthetic_df = pd.read_csv(synthetic_path)

    logger.info(f"Real data shape: {real_df.shape}")
    logger.info(f"Synthetic data shape: {synthetic_df.shape}")

    if output_file is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"llm_results_{provider}_{ts}.jsonl"

    all_results = []
    trial_id = 0

    # Build trial queue: balanced real/synthetic per condition
    trial_queue = []
    for condition in conditions:
        for _ in range(n_trials // 2):
            trial_queue.append(("REAL", condition))
            trial_queue.append(("SYNTHETIC", condition))

    random.shuffle(trial_queue)

    with open(output_file, "w") as f:
        for true_label, condition in tqdm(trial_queue, desc=f"{provider} / {model}"):
            df = real_df if true_label == "REAL" else synthetic_df

            # Optionally sample a subset of rows for variety
            if sample_rows and sample_rows < len(df):
                df = df.sample(n=sample_rows, random_state=trial_id).reset_index(drop=True)

            result = run_trial(
                df=df,
                true_label=true_label,
                condition=condition,
                provider=provider,
                model=model,
                trial_id=trial_id,
                n_rows_shown=n_rows_shown,
            )

            f.write(json.dumps(result) + "\n")
            f.flush()
            all_results.append(result)

            logger.info(
                f"Trial {trial_id} | {condition} | label={true_label} | "
                f"pred={result['predicted_label']} | correct={result['correct']} | "
                f"conf={result['confidence']}"
            )
            trial_id += 1
            time.sleep(0.5)  # polite rate limiting

    # Summary stats
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby(["condition", "true_label"])["correct"].agg(["sum", "count", "mean"])
    logger.info(f"\n=== SUMMARY ===\n{summary}")

    summary_path = str(output_file).replace(".jsonl", "_summary.csv")
    results_df.to_csv(summary_path, index=False)
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Summary saved to {summary_path}")

    return all_results



# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM discrimination experiment (single or multi-method)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (Multi-method mode):
  # Auto-discover all methods, use OpenAI:
  python evaluate.py --real data/real/adult.csv --provider openai

  # Specific methods with Google Gemini (free & fast!):
  python evaluate.py --real data/real/adult.csv --methods ctgan tvae --provider google --model gemini-2.5-flash

  # Test multiple Google models in sequence (Phase 2 experiments):
  python evaluate.py --real data/real/adult.csv --provider google \\
    --compare-models gemini-2.5-flash gemini-2.5-pro gemini-2.0-flash gemini-3-flash-preview

  # Groq Llama 3.1 (free & generous rate limits):
  python evaluate.py --real data/real/adult.csv --provider groq --model llama-3.1-70b-versatile --n_trials 5

  # GitHub Models GPT-4o (free):
  python evaluate.py --real data/real/adult.csv --provider github --model gpt-4o

FREE API Providers:
  --provider google  (Gemini models - 1500 req/day, fastest)
  --provider github  (GPT-4o, Llama 3.3, etc - 150 req/day)
  --provider groq    (Llama 3.1, Mixtral - 14400 req/day, fastest speed)

See FREE_API_SETUP.md for setup instructions!
        """
    )
    
    parser.add_argument("--real", required=True, help="Path to real data CSV")
    parser.add_argument("--synthetic", default=None, help="Path to synthetic CSV (legacy single-method mode)")
    parser.add_argument("--synthetic-dir", default="data/synthetic", help="Directory containing synthetic method folders")
    parser.add_argument("--methods", nargs="+", default=None, help="Specific methods to run (auto-discovers if not specified)")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "github", "google", "groq"],
        help="LLM provider (openai | anthropic | github | google | groq)"
    )
    parser.add_argument("--model", default="gpt-4o", help="Single model to test")
    parser.add_argument("--compare-models", nargs="+", default=None, help="Test multiple models in sequence (e.g., gemini-2.5-flash gemini-2.5-pro)")
    parser.add_argument("--n_trials", type=int, default=10, help="Trials per label per condition")
    parser.add_argument("--conditions", nargs="+", default=["C1", "C2"])
    parser.add_argument("--n_rows_shown", type=int, default=20)
    parser.add_argument("--sample_rows", type=int, default=None)
    args = parser.parse_args()

    # Determine which models to test
    models_to_test = args.compare_models if args.compare_models else [args.model]
    
    # Test each model
    for test_model in models_to_test:
        logger.info(f"\n{'='*70}")
        logger.info(f"TESTING MODEL: {test_model}")
        logger.info(f"{'='*70}\n")
        
        if args.synthetic:
            # Legacy single-method mode
            logger.info("Running in legacy single-method mode...")
            run_experiment_legacy(
                real_path=args.real,
                synthetic_path=args.synthetic,
                provider=args.provider,
                model=test_model,
                n_trials=args.n_trials,
                conditions=args.conditions,
                n_rows_shown=args.n_rows_shown,
                sample_rows=args.sample_rows,
            )
        else:
            # Multi-method mode
            logger.info("Running in multi-method mode...")
            run_experiment(
                real_path=args.real,
                synthetic_dir=args.synthetic_dir,
                synthetic_methods=args.methods,
                provider=args.provider,
                model=test_model,
                n_trials=args.n_trials,
                conditions=args.conditions,
                n_rows_shown=args.n_rows_shown,
                sample_rows=args.sample_rows,
            )
        
        # Brief pause between models
        if test_model != models_to_test[-1]:
            logger.info(f"Pausing 30s before next model...")
            time.sleep(30)

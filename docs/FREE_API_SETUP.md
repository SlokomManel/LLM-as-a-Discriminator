# Free API Keys Setup Guide

You can now use multiple free LLM providers instead of paid OpenAI/Anthropic APIs. Here's how to set up each:

## 1. **GitHub Models** (⭐ Best for GPT-4o)

**Models Available:** GPT-4o, Llama 3.3, DeepSeek-R1, Mistral  
**Limits:** ~10-15 requests/min, ~150 requests/day  
**Cost:** Free

### Setup:
1. Go to https://github.com/marketplace/models
2. Sign in with your GitHub account (create one if needed)
3. Select a model (e.g., gpt-4o)
4. Create a **Personal Access Token (PAT)**:
   - Go to GitHub Settings → Developer Settings → Personal Access Tokens
   - Create a new token with `repo` scope
5. Set environment variable:
   ```bash
   export GITHUB_PAT="ghp_xxxxxxxxxxxx"
   ```

### Run trials:
```bash
python code/evaluate.py --real data/real/adult.csv --provider github --model gpt-4o --n_trials 3
```

---

## 2. **Google AI Studio** (⭐ Best for high volume)

**Models Available:** Gemini 1.5 Flash, Gemini 1.5 Pro  
**Limits:** 1,500 requests/day, 15 requests/min  
**Cost:** Free

### Setup:
1. Go to https://aistudio.google.com/app/apikey
2. Click **"Create API Key"**
3. Copy the key
4. Set environment variable:
   ```bash
   export GOOGLE_API_KEY="AIzaSy_xxxxxxxxxxxx"
   ```

### Run trials:
```bash
python code/evaluate.py --real data/real/adult.csv --provider google --model gemini-1.5-flash --n_trials 3
```

**Note:** Gemini 1.5 Flash is extremely fast and great for testing!

---

## 3. **Groq Cloud** (⭐ Best for speed + volume)

**Models Available:** Llama 3.1 (8B, 70B, 405B), Mixtral, Gemma  
**Limits:** ~14,400 requests/day (very generous!)  
**Cost:** Free

### Setup:
1. Go to https://console.groq.com/keys
2. Sign up or sign in (GitHub/Google/Email)
3. Click **"Create API Key"**
4. Copy the key
5. Set environment variable:
   ```bash
   export GROQ_API_KEY="gsk_xxxxxxxxxxxx"
   ```

### Run trials:
```bash
python code/evaluate.py --real data/real/adult.csv --provider groq --model llama-3.1-70b-versatile --n_trials 3
```

---

## Quick Comparison

| Provider | Best For | Setup Time | Speed | Cost |
|----------|----------|-----------|-------|------|
| **GitHub** | GPT-4o free tier | 5 min | Medium | Free |
| **Google** | High volume (1.5k/day) | 2 min | ⚡ Very Fast | Free |
| **Groq** | Extreme volume (14k+/day) | 3 min | ⚡⚡ Fastest | Free |
| **OpenAI** | Production quality | 2 min | Medium | $$ Paid |
| **Anthropic** | Claude fallback | 2 min | Medium | $$ Paid |

---

## Batch Setup (All at once)

If you want to set up all free APIs at once:

```bash
# Create a .env file
cat > /Users/mslokom/Documents/research_directions/data-synthesis-sprite/PSD2026/.env << 'EOF'
# Free APIs
GITHUB_PAT="ghp_xxxxxxxxxxxx"
GOOGLE_API_KEY="AIzaSy_xxxxxxxxxxxx"
GROQ_API_KEY="gsk_xxxxxxxxxxxx"

# Paid APIs (optional)
OPENAI_API_KEY="sk-proj-..."
ANTHROPIC_API_KEY="sk-ant-..."

# Results directory
RESULTS_DIR="results"
EOF
```

Then load it automatically (already in code!):
```bash
# No need to export manually - evaluate.py loads .env automatically
python code/evaluate.py --real data/real/adult.csv --provider google --n_trials 3
```

---

## Run Full Multi-Provider Experiment

Test all providers at once:

```bash
# OpenAI (if quota fixed)
python code/evaluate.py --real data/real/adult.csv --provider openai --n_trials 2

# GitHub Models (free GPT-4o)
python code/evaluate.py --real data/real/adult.csv --provider github --model gpt-4o --n_trials 2

# Google (fastest)
python code/evaluate.py --real data/real/adult.csv --provider google --model gemini-1.5-flash --n_trials 2

# Groq (extreme volume)
python code/evaluate.py --real data/real/adult.csv --provider groq --model llama-3.1-70b-versatile --n_trials 2
```

---

## Experiment Best Practices Applied

✅ **Temperature=0**: All providers use temperature=0 for consistent reasoning  
✅ **Markdown Tables**: Dataframes formatted as readable Markdown  
✅ **Row Sampling**: Large tables sampled to 20 rows for efficiency  
✅ **Label Shuffling**: Dataset identities randomized to reduce bias  
✅ **Error Handling**: Automatic retry on API failures  
✅ **Structured Output**: All responses in JSON format for analysis  

---

## Recommended First Trial

Start with **Google Gemini 1.5 Flash** (fastest, most generous limits):

```bash
export GOOGLE_API_KEY="AIzaSy_xxxxxxxxxxxx"
python code/evaluate.py --real data/real/adult.csv --provider google --model gemini-1.5-flash --n_trials 5
```

Expected time: <2 minutes  
Cost: Free  
Output: Results in `results/llm_results_*.jsonl`

Then compare with other providers if needed!

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `API Key not found` | Make sure to `export` the key or add to `.env` |
| `Rate limit exceeded` | Wait a minute, reduce `--n_trials`, or use different provider |
| `JSON parsing error` | Some models are bad at JSON; try different provider |
| `Module not found` | Run: `.venv/bin/pip install google-generativeai groq` |

---

## Next Steps

1. Pick a free provider above
2. Get an API key (2-5 minutes)
3. Set the environment variable
4. Run pilot trials: `python code/evaluate.py --real data/real/adult.csv --provider [google|github|groq] --n_trials 3`
5. Check results in `results/` directory
6. Compare providers and pick the best one for your budget/speed needs!

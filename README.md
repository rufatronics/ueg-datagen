# UEG Training Data Generator

Automated pipeline that generates the 176,000+ labeled training examples needed for the Universal Edge Gateway (UEG) intent classifier.

## Architecture

- **Groq** — volume engine for Tier 1–4 English classes (4 models in parallel, ~4K req/day each)
- **Mistral** — Tier 5A + 5B English classes (1B token/month budget)
- **Gemini Flash-Lite** — Tier 5B English overflow (1,000 req/day)
- **Gemini Flash** — all non-English languages, lr_emerging, mul_mix (250 req/day)
- **Gemini Pro** — hardest combos: lr_emerging + Tier 5B (100 req/day)
- **Class 02 (adversarial_probe)** — pulled from existing public red-team HF datasets

## Setup

### 1. Fork / clone this repo (keep it PUBLIC — free unlimited Actions minutes)

### 2. Add GitHub Secrets

Go to **Settings → Secrets and variables → Actions → New repository secret**

| Secret name     | Value |
|----------------|-------|
| `GROQ_API_KEY`  | your Groq key |
| `GEMINI_API_KEY`| your Google AI Studio key |
| `MISTRAL_API_KEY`| your Mistral key |
| `HF_TOKEN`      | your HuggingFace access token |

### 3. Enable GitHub Actions

Go to **Actions tab → Enable workflows**

### 4. Trigger first run manually

Actions → **Generate UEG Training Data** → **Run workflow**

The pipeline will then run automatically every 2 hours via cron.

## How it works

1. On every run, state is loaded from HuggingFace (`progress.json`)
2. Generator fills each class toward 8,000 examples using the provider rotation
3. Every 50 examples → checkpoint pushed to HF + state saved
4. On clean exit, full state saved — next run resumes exactly where this one stopped
5. When a class hits 8,000 examples it's marked complete and never written to again
6. When all 22 classes are complete, future runs exit immediately

## Stopping

To pause generation: create a file named `STOP` in the repo root and push it.
To resume: delete the `STOP` file.

## Output

Dataset at: `https://huggingface.co/datasets/rufatronics/ueg-training-data`

One JSONL file per class:
```
data/class_01_noise_gibberish.jsonl
data/class_02_adversarial_probe.jsonl
...
data/class_22_multilingual_task.jsonl
```

Each line:
```json
{
  "text": "...",
  "intent_class_id": 13,
  "intent_class_label": "code_task",
  "tier": "5A",
  "language_iso": "en",
  "resource_class": "hr_global",
  "generated_by": "groq:llama-3.3-70b-versatile",
  "split": "train"
}
```

## Monitoring

Check daily status: Actions → **Generation Status Report** → **Run workflow**

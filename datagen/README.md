# UEG Training Data Generator v2

Automated pipeline generating 176,000+ labeled examples for the Universal Edge Gateway intent classifier.

## Architecture

| Provider | Models | Role | Batch |
|----------|--------|------|-------|
| **Groq** | llama-3.1-8b, llama-3.3-70b, llama-4-scout | Standard English classes, volume | 20/call |
| **Gemini (Gemma 4)** | gemma-4-26b, gemma-4-31b | Mixed-class, unlimited TPM | 80/call |
| **Gemini (Flash)** | gemini-3.1-flash-lite, gemini-3.5-flash | Supplemental | 20/call |
| **Mistral** | mistral-large, mistral-small | Tier 5A/5B overflow | 15/call |

Groq + Gemini run in **parallel threads** simultaneously.

## Language distribution
70% English + 30% Arabic, Hindi, French, Spanish, Chinese, Swahili, Portuguese — enough for the classifier to learn intent is language-agnostic.

## Setup

### 1. Create a public GitHub repo and push all files (keep public — free unlimited Actions minutes)

### 2. Add secrets (Settings → Secrets → Actions)
| Secret | Value |
|--------|-------|
| `GROQ_API_KEY` | your Groq key |
| `GEMINI_API_KEY` | your Google AI Studio key |
| `MISTRAL_API_KEY` | your Mistral key |
| `HF_TOKEN` | your HuggingFace token |

### 3. Enable Actions and trigger first run manually

Runs automatically every 4 hours after that.

## How it works
1. Reads `progress.json` from HuggingFace on every startup — resumes exactly where it left off
2. Groq + Gemini generate in parallel, Mistral fills remaining budget
3. Every 50 examples → checkpoint pushed to HF + state saved
4. Completed classes never written to again
5. All done → exits cleanly, future runs no-op

## Controls
- **Pause:** create a file named `STOP` in repo root and push it
- **Resume:** delete the `STOP` file
- **Monitor:** Actions → Daily Status Report → Run workflow

## Output
`https://huggingface.co/datasets/rufatronics/ueg-training-data`

Per-class JSONL files + `progress.json` state.

# UEG Benchmark

Compares UEG vs Regex vs Llama-3.1-8B across 50 test cases in 6 categories.

## Run locally

```bash
pip install requests huggingface_hub groq

export HF_TOKEN="your_hf_token"
export GROQ_API_KEY="your_groq_key"

python benchmark.py
```

## Categories

- **easy** — clean well-formed English, all three should handle these
- **ambiguous** — same words, different intent — where regex breaks
- **noisy** — typos, ALL CAPS, Nigerian Pidgin, code-switched
- **non_english** — Arabic, French, Spanish, Hindi, Swahili, Chinese, Portuguese — regex has zero coverage
- **adversarial** — creative jailbreaks without obvious keywords
- **tier5** — complex Tier 5 inputs regex can't pattern match

## Results pushed to

https://huggingface.co/datasets/rufatronics/ueg-benchmark-results

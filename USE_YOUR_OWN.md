# Build Your Own Intent Classifier

This guide shows you how to use the UEG pipeline to build a completely custom intent classifier for your own use case — different classes, different languages, different deployment.

Everything is open source and free. You need:
- A HuggingFace account (free)
- A GitHub account (free)
- A Kaggle account (free)
- API keys for at least one provider (Groq free tier works)

---

## Step 1 — Fork the Repos

Fork these two repos to your GitHub account:

- **API**: [github.com/rufatronics/ueg-api](https://github.com/rufatronics/ueg-api)
- **Data + Training**: [github.com/rufatronics/ueg-training-data](https://github.com/rufatronics/ueg-training-data) *(this repo)*

---

## Step 2 — Define Your Taxonomy

The taxonomy is the most important decision. Think about:
- What actions does your system need to take?
- What's the minimum number of classes that maps to different routing decisions?
- What languages do your users speak?

Example taxonomy for a customer support bot:

```python
INTENT_CLASSES = {
    1:  "spam_noise",           # Tier 1 — drop
    2:  "abuse_violation",      # Tier 1 — block
    3:  "greeting",             # Tier 2 — template
    4:  "farewell",             # Tier 2 — template
    5:  "order_status",         # Tier 3 — database lookup
    6:  "return_request",       # Tier 3 — database lookup
    7:  "billing_question",     # Tier 4 — cached FAQ
    8:  "product_question",     # Tier 4 — cached FAQ
    9:  "complaint",            # Tier 5 — agent
    10: "technical_support",    # Tier 5 — agent
    11: "escalation_request",   # Tier 5 — human handoff
}
```

Aim for 8-25 classes. Too few and you lose routing precision. Too many and you need more training data.

---

## Step 3 — Generate Training Data

The data generation pipeline uses free LLM APIs to generate labeled examples automatically.

### Setup

```bash
git clone https://github.com/your-username/ueg-training-data
cd ueg-training-data
pip install -r datagen/requirements.txt
```

### Configure

Edit `datagen/src/taxonomy.py` with your classes. Each class needs:
- A label name
- A tier
- A description for the prompt
- Whether it's English-only or multilingual

### Get free API keys

| Provider | Free Tier | Models |
|----------|-----------|--------|
| [Groq](https://console.groq.com) | 14K req/day | Llama 3.1 8B, 70B |
| [Google AI Studio](https://aistudio.google.com) | 1,500 req/day | Gemma 4, Gemini Flash |
| [Mistral](https://console.mistral.ai) | Free tier | Ministral, Mistral Small |

### Add secrets to GitHub

In your forked repo → Settings → Secrets → Actions:
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `MISTRAL_API_KEY`
- `HF_TOKEN`

### Run the pipeline

Go to Actions → Generate Training Data → Run workflow.

It runs every 4 hours automatically and pushes data to your HuggingFace dataset repo.

**Target:** 4,000-8,000 examples per class minimum. More is better.

---

## Step 4 — Create HuggingFace Repos

```python
from huggingface_hub import HfApi
api = HfApi(token="your_token")

# Dataset repo for training data
api.create_repo("your-username/your-classifier-data", repo_type="dataset")

# Model repo for trained weights
api.create_repo("your-username/your-classifier", repo_type="model")
```

Update the repo names in `datagen/src/state.py` and the training notebook.

---

## Step 5 — Train on Kaggle

1. Go to [kaggle.com](https://kaggle.com) → Create Notebook
2. Upload `training/ueg_training_v3.ipynb`
3. In Cell 3, update:
   ```python
   CFG.data_repo  = "your-username/your-classifier-data"
   CFG.model_repo = "your-username/your-classifier"
   CFG.num_intent_classes = len(YOUR_INTENT_CLASSES)
   ```
4. In Cell 4, replace `INTENT_CLASSES` with your taxonomy
5. Add `HF_TOKEN` as a Kaggle secret
6. Save & Run All — go to sleep

The notebook handles everything: tokenizer training, model training, evaluation, ONNX export, upload to HF.

---

## Step 6 — Deploy the API

```bash
git clone https://github.com/your-username/ueg-api
cd ueg-api
```

Edit `config.py`:
```python
MODEL_REPO = "your-username/your-classifier"

INTENT_CLASSES = {
    # your classes here
}

ROUTING_ACTIONS = {
    # your routing logic here
}
```

Deploy to Render:
1. Push to GitHub
2. render.com → New Web Service → Connect repo
3. It reads `render.yaml` automatically
4. Deploy

---

## Step 7 — Run the Benchmark

```bash
cd benchmark
pip install requests groq
HF_TOKEN=your_token GROQ_API_KEY=your_key python benchmark.py
```

Edit `benchmark/benchmark.py` to point at your endpoint and update the test cases for your taxonomy.

---

## Architecture Reference

The model you'll train is identical to UEG:

```
Input text
    ↓
Custom BPE Tokenizer (32K vocab, trained on your data)
    ↓
6x Transformer Blocks (512 hidden, 8 heads, 2048 FFN)
    ↓
[CLS] token
    ↙           ↘
Head A          Head B
N-class         5-class
intent          resource density
```

35M parameters. Trains in 3-6 hours on Kaggle free GPU. Exports to ONNX for <5ms inference.

---

## HuggingFace Resources

| Resource | Link |
|----------|------|
| Base Model (to fine-tune from) | [rufatronics/ueg-classifier](https://huggingface.co/rufatronics/ueg-classifier) |
| Example Training Data | [rufatronics/ueg-training-data](https://huggingface.co/datasets/rufatronics/ueg-training-data) |
| Benchmark Results | [rufatronics/ueg-benchmark-results](https://huggingface.co/datasets/rufatronics/ueg-benchmark-results) |

---

## Cost Summary

Everything in this pipeline can be done for **$0**:

| Step | Tool | Cost |
|------|------|------|
| Data generation | Groq/Gemini/Mistral free tiers | $0 |
| Training | Kaggle free GPU | $0 |
| Model hosting | HuggingFace free | $0 |
| API hosting | Render free tier | $0 |
| Benchmark | Groq free tier | $0 |

The only cost is time. A full pipeline from taxonomy definition to deployed API takes about 2-3 days.

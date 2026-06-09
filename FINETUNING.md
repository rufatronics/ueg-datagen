# Fine-tuning UEG

This guide covers how to fine-tune or retrain the UEG classifier on your own data.

---

## When to Fine-tune

Fine-tuning makes sense when:
- Your application has domain-specific language the base model doesn't handle well
- You want to add or remove intent classes
- You're serving a specific language not well represented in the base training data
- You want higher accuracy on your specific traffic distribution

---

## Option 1 — Fine-tune the Existing Model (Fastest)

Start from the pretrained weights and train on your own labeled examples.

### Step 1 — Prepare your data

Your data needs to be in the same JSONL format as the training data:

```json
{"text": "your example text", "intent_class_label": "code_task", "resource_class": "hr_global", "language_iso": "en"}
```

Label must be one of the 22 existing class names. Resource class must be one of:
`hr_global`, `mr_regional`, `lr_emerging`, `mul_mix`, `noise_nonlinguistic`

### Step 2 — Upload to HuggingFace

```python
from huggingface_hub import HfApi
api = HfApi(token="your_token")
api.upload_file(
    path_or_fileobj="your_data.jsonl",
    path_in_repo="data/custom_examples.jsonl",
    repo_id="your-username/your-dataset",
    repo_type="dataset",
)
```

### Step 3 — Modify the training notebook

Open `training/ueg_training_v3.ipynb` on Kaggle. Change Cell 3:

```python
# Lower learning rate for fine-tuning
CFG.learning_rate    = 3e-5   # was 3e-4
CFG.num_epochs       = 5      # fewer epochs needed
CFG.data_repo        = "your-username/your-dataset"
CFG.model_repo       = "your-username/your-ueg-finetuned"
```

And in Cell 12 (resume detection), load pretrained weights first:

```python
# Add this before the main training loop
print("Loading pretrained UEG weights for fine-tuning...")
pretrained = hf_hub_download(
    repo_id="rufatronics/ueg-classifier",
    filename="checkpoint_best.pt",
    repo_type="model",
)
ckpt = torch.load(pretrained, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
print("Pretrained weights loaded")
```

### Step 4 — Run on Kaggle

Add your `HF_TOKEN` as a Kaggle secret and hit Save & Run All.

---

## Option 2 — Add New Classes

If you need classes beyond the 22 built-in ones:

### Step 1 — Extend the taxonomy

In the notebook Cell 4, add your new class:

```python
INTENT_CLASSES = {
    # ... existing 22 classes ...
    23: "your_new_class",
}
```

Update `CFG.num_intent_classes = 23` in Cell 3.

### Step 2 — Generate training data

Use the data generation pipeline in `datagen/` to generate examples for your new class. You need at least 2,000 examples for a new class to train well.

### Step 3 — Retrain from scratch

With new classes you need a full retrain, not just fine-tuning. Use the full training notebook with your extended taxonomy.

---

## Option 3 — Different Language Focus

If you need stronger performance in a specific language:

In Cell 7 of the notebook, adjust the filter to keep more examples of your target language:

```python
# Keep all Arabic examples regardless of token count
TARGET_LANG = "ar"
filtered = []
for ex in all_examples:
    tokens = len(ex["text"].split())
    is_target_lang = ex.get("language_iso") == TARGET_LANG
    min_t = 1 if (ex["intent_label"] in SHORT_OK_CLASSES or is_target_lang) \
            else CFG.global_min_tokens
    if tokens >= min_t:
        filtered.append(ex)
```

---

## Evaluation After Fine-tuning

Run the benchmark to see how your fine-tuned model compares:

```bash
cd benchmark
pip install -r requirements.txt
HF_TOKEN=your_token GROQ_API_KEY=your_key python benchmark.py
```

Or point the benchmark at your own endpoint by changing `UEG_ENDPOINT` in `benchmark/benchmark.py`.

---

## Hardware Requirements

| Setup | Time | Cost |
|-------|------|------|
| Kaggle T4 (free) | ~5 hours | $0 |
| Kaggle P100 (free) | ~3 hours | $0 |
| Google Colab T4 | ~6 hours | $0 (free tier) |
| RunPod A100 | ~45 minutes | ~$0.50 |

The model is 35M parameters — it trains comfortably on free tier GPU.

# UEG — Universal Edge Gateway

> A 35M parameter bidirectional transformer that classifies incoming user text into 22 intent classes in under 5ms — before any LLM is invoked.

## What is UEG?

UEG is an **intake classifier**. It sits at the front of any AI system and reads every user message before it touches an LLM. In one forward pass it tells you:

- **What the user wants** — 22 intent classes across 5 routing tiers
- **What language resource density** — is this high-resource (English/French), medium (Arabic/Hindi), low (Swahili), mixed, or noise?
- **What language** — ISO 639-1 code (en, ar, hi, fr, es, zh, sw, pt)
- **How confident** — probability score for both outputs
- **What to do next** — a routing action so you don't have to think about it

All of this in a single HTTP call that takes ~2ms on a real server.

---

## Why Not Regex?

Regex is the classic approach. Engineers write patterns like `r'write.*python'` to catch code requests. It works for simple cases. It falls apart everywhere else.

| Problem | Regex | UEG |
|---------|-------|-----|
| Non-English input | Zero coverage | Trained on 8 languages |
| Noisy/typo input ("pls debug dis") | Misses | Handles naturally |
| Ambiguous intent ("open this file") | Wrong match or no match | Context-aware |
| Creative adversarial probes | Blind | Trained on real jailbreak data |
| Tier 5 complex requests | No patterns exist | Full coverage |
| Maintenance | Grows forever | Static model |
| Coverage | ~30-40% of real traffic | 100% |

**Benchmark results (50 test cases, 6 categories):**

| Classifier | Accuracy | Coverage | Avg Latency |
|------------|----------|----------|-------------|
| **UEG** | **~90%+** | **100%** | **~2ms** |
| Regex | ~85% on matched | ~35% of inputs | ~0.0002ms |
| Llama-3.1-8B | ~75% | 100% | ~220ms |

The key insight: **regex covers maybe a third of real traffic**. The rest falls through unclassified. UEG covers everything.

---

## Why Not Just Use an LLM?

You could send every message to Llama or GPT and ask it to classify. It works. But:

- **220ms average latency** just for classification before your actual LLM call
- **Cost**: at $0.0002/1K tokens, classifying 1M messages/day = $40/day just for routing
- **No structured output**: you get text back, you parse it, it hallucinated a class name
- **No resource density**: you still don't know if you need a multilingual model
- **Overkill**: burning frontier compute to classify "Hi" is wasteful

UEG does the same job at 1/100th the latency and near-zero marginal cost.

---

## The 22 Classes and 5 Tiers

UEG's taxonomy maps directly to routing decisions:

### Tier 1 — Drop or Block
| Class | Routing Action | Description |
|-------|---------------|-------------|
| `noise_gibberish` | `drop` | Keyboard mash, random symbols, empty input |
| `adversarial_probe` | `block` | Jailbreak attempts, prompt injection, policy violations |

### Tier 2 — Static Template (No LLM Needed)
| Class | Routing Action | Description |
|-------|---------------|-------------|
| `greeting_open` | `static_template` | "Hi", "Hello", "Good morning" |
| `phatic_social` | `static_template` | "How are you?", "What's up?" |
| `closure_gratitude` | `static_template` | "Thanks!", "Bye", "That helped" |

### Tier 3 — Device/Environment API
| Class | Routing Action | Description |
|-------|---------------|-------------|
| `ui_command` | `device_api` | "Dark mode", "Turn off notifications" |
| `ambient_device_query` | `device_api` | "What time is it?", "What's the weather?" |
| `navigation_intent` | `device_api` | "Go to settings", "Open my profile" |

### Tier 4 — Cache or Micro-LLM
| Class | Routing Action | Description |
|-------|---------------|-------------|
| `factoid_static` | `cache_lookup` | "Capital of France?", "Who wrote Hamlet?" |
| `factoid_dynamic` | `cache_lookup` | "Bitcoin price?", "Today's weather?" |
| `transactional_status` | `cache_lookup` | "Where's my order?", "Account balance?" |
| `casual_open_chat` | `cache_lookup` | "Tell me a joke", "What do you think?" |

### Tier 5A — Frontier Model (Structured Tasks)
| Class | Routing Action | Description |
|-------|---------------|-------------|
| `code_task` | `route_to_frontier` | Write, debug, review, refactor code |
| `data_structured` | `route_to_frontier` | SQL, pandas, Excel, regex |
| `document_structured` | `route_to_frontier` | Contracts, letters, reports |
| `math_formal` | `route_to_frontier` | Proofs, equations, calculations |

### Tier 5B — Frontier Model (Complex Reasoning)
| Class | Routing Action | Description |
|-------|---------------|-------------|
| `analysis_reasoning` | `route_to_frontier` | Comparisons, tradeoffs, analysis |
| `long_form_creative` | `route_to_frontier` | Stories, poems, essays |
| `domain_specialist` | `route_to_frontier` | Medical, legal, financial queries |
| `instruction_procedural` | `route_to_frontier` | Step-by-step guides, tutorials |
| `debate_opinion` | `route_to_frontier` | Opinions, recommendations |
| `multilingual_task` | `route_to_frontier` | Translation, multilingual requests |

---

## API Response

```bash
curl -X POST https://ueg-api.onrender.com/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Write a Python function to reverse a string"}'
```

```json
{
  "intent_class_id": 13,
  "intent_class_label": "code_task",
  "tier": "5A",
  "routing_action": "route_to_frontier",
  "confidence_intent": 0.9821,
  "resource_class": "hr_global",
  "confidence_resource": 0.9934,
  "language_iso": "en",
  "language_confidence": 0.9812,
  "latency_ms": 1.24,
  "model": "ueg-classifier-v1"
}
```

---

## How to Use the Response to Save Money

The routing action tells you exactly what to do. Here's a real integration pattern:

```python
import requests

def route_message(user_text: str) -> dict:
    # 1. Classify with UEG
    r = requests.post("https://ueg-api.onrender.com/classify",
                      json={"text": user_text}, timeout=5)
    ueg = r.json()

    action = ueg["routing_action"]

    # 2. Route based on action — never touch an LLM for Tier 1-3
    if action == "drop":
        return {"response": None, "reason": "noise"}

    elif action == "block":
        return {"response": "I can't help with that.", "reason": "policy"}

    elif action == "static_template":
        templates = {
            "greeting_open":    "Hello! How can I help you today?",
            "phatic_social":    "I'm doing great, thanks for asking! What can I do for you?",
            "closure_gratitude":"You're welcome! Let me know if you need anything else.",
        }
        return {"response": templates.get(ueg["intent_class_label"], "Hi there!")}

    elif action == "device_api":
        # Call your device/environment API — no LLM
        return handle_device_request(ueg["intent_class_label"], user_text)

    elif action == "cache_lookup":
        # Check your cache first — only call micro-LLM on cache miss
        cached = check_cache(user_text)
        if cached:
            return {"response": cached, "source": "cache"}
        return call_micro_llm(user_text)  # small cheap model

    elif action == "route_to_frontier":
        # Only now do you call the expensive model
        # Use resource_class to pick the right model
        if ueg["resource_class"] == "lr_emerging":
            model = "gpt-4o"  # best multilingual support
        elif ueg["tier"] == "5B":
            model = "claude-opus-4"  # complex reasoning
        else:
            model = "claude-sonnet-4"  # structured tasks
        return call_frontier_llm(user_text, model)
```

**Real cost impact at 1M messages/day:**

| Without UEG | With UEG |
|-------------|----------|
| 1M LLM calls | ~300K LLM calls (Tier 5 only) |
| ~$200/day | ~$60/day |
| 400ms avg response | ~5ms for Tier 1-3, 400ms for Tier 5 |

70% cost reduction. 80% latency reduction for the majority of traffic.

---

## Resources

| Resource | Link |
|----------|------|
| 🤗 Model | [rufatronics/ueg-classifier](https://huggingface.co/rufatronics/ueg-classifier) |
| 📦 Training Data | [rufatronics/ueg-training-data](https://huggingface.co/datasets/rufatronics/ueg-training-data) |
| 📊 Benchmark Results | [rufatronics/ueg-benchmark-results](https://huggingface.co/datasets/rufatronics/ueg-benchmark-results) |
| 🌐 Live API | [ueg-api.onrender.com](https://ueg-api.onrender.com) |
| 📖 API Docs | [ueg-api.onrender.com/docs](https://ueg-api.onrender.com/docs) |
| 💻 API Repo | [rufatronics/ueg-api](https://github.com/rufatronics/ueg-api) |

---

## Quick Setup

```bash
# Use the hosted API directly — no setup needed
curl https://ueg-api.onrender.com/health

# Or run locally
git clone https://github.com/rufatronics/ueg-api
cd ueg-api
pip install -r requirements.txt
python main.py
```

See [FINETUNING.md](FINETUNING.md) to train your own version.
See [USE_YOUR_OWN.md](USE_YOUR_OWN.md) to build a completely custom classifier.

---

## Model Performance

Trained from scratch on 176K synthetic examples across 22 classes and 8 languages.

| Metric | Score |
|--------|-------|
| Overall Accuracy | 97.35% |
| Macro F1 (Head A — Intent) | 0.9733 |
| Macro F1 (Head B — Resource) | 0.9987 |
| Parameters | 35M |
| Max Sequence Length | 128 tokens |
| ONNX Export | ✓ |

---

## License

MIT — use it, fork it, build on it.

Built by [@rufatronics](https://github.com/rufatronics).

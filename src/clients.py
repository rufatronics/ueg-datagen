"""
API clients for all providers.
Each client handles rate limiting, retries, and error logging internally.
"""

import os
import time
import json
import requests
from typing import Optional

GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]
HF_TOKEN        = os.environ["HF_TOKEN"]


# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------

def _post_with_retry(url: str, headers: dict, payload: dict,
                     provider: str, max_retries: int = 3) -> Optional[str]:
    """POST with exponential backoff. Returns response text or None."""
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)

            if r.status_code == 200:
                return r.text

            elif r.status_code == 429:
                wait = (2 ** attempt) * 10
                print(f"[{provider}] Rate limited (429) — waiting {wait}s")
                time.sleep(wait)

            elif r.status_code in (500, 502, 503, 504):
                wait = (2 ** attempt) * 5
                print(f"[{provider}] Server error {r.status_code} — waiting {wait}s")
                time.sleep(wait)

            elif r.status_code == 401:
                print(f"[{provider}] Auth error (401) — check API key")
                return None

            elif r.status_code == 400:
                print(f"[{provider}] Bad request (400): {r.text[:300]}")
                return None

            else:
                print(f"[{provider}] Unexpected status {r.status_code}: {r.text[:200]}")
                time.sleep(5)

        except requests.exceptions.Timeout:
            print(f"[{provider}] Timeout on attempt {attempt+1}")
            time.sleep(10)
        except requests.exceptions.ConnectionError as e:
            print(f"[{provider}] Connection error: {e}")
            time.sleep(15)
        except Exception as e:
            print(f"[{provider}] Unexpected error: {e}")
            time.sleep(5)

    print(f"[{provider}] All {max_retries} attempts failed")
    return None


def _extract_text_content(response_text: str, provider: str) -> Optional[str]:
    """Extract the text content from a provider response JSON."""
    try:
        data = json.loads(response_text)
    except Exception as e:
        print(f"[{provider}] Response parse error: {e}")
        return None

    try:
        if provider == "groq" or provider == "mistral":
            return data["choices"][0]["message"]["content"]
        elif provider.startswith("gemini"):
            return data["candidates"][0]["content"]["parts"][0]["text"]
        elif provider == "openrouter":
            return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"[{provider}] Content extraction error: {e} | Keys: {list(data.keys())}")
        return None


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def call_groq(prompt: str, model: str) -> Optional[str]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 4096,
    }
    raw = _post_with_retry(url, headers, payload, "groq")
    if raw is None:
        return None
    return _extract_text_content(raw, "groq")


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, model: str) -> Optional[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 8192,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }
    raw = _post_with_retry(url, headers, payload, f"gemini/{model}")
    if raw is None:
        return None
    return _extract_text_content(raw, "gemini")


# ---------------------------------------------------------------------------
# Mistral client
# ---------------------------------------------------------------------------

def call_mistral(prompt: str, model: str) -> Optional[str]:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 4096,
    }
    raw = _post_with_retry(url, headers, payload, "mistral")
    if raw is None:
        return None
    return _extract_text_content(raw, "mistral")


# ---------------------------------------------------------------------------
# Adversarial data — pulled from public HuggingFace red-team datasets
# ---------------------------------------------------------------------------

def fetch_adversarial_from_hf(max_examples: int = 500) -> list[dict]:
    """
    Pull from existing public adversarial/jailbreak datasets on HuggingFace.
    Reformats into UEG schema. Far better quality than synthetic generation.
    Sources: jailbreak_llms, advbench, do-not-answer
    """
    results = []

    sources = [
        {
            "url": "https://datasets-server.huggingface.co/rows?dataset=verazuo%2Fjailbreak_llms&config=default&split=train&offset=0&length=100",
            "text_field": "prompt",
        },
        {
            "url": "https://datasets-server.huggingface.co/rows?dataset=walledai%2FAdvBench&config=default&split=train&offset=0&length=100",
            "text_field": "prompt",
        },
        {
            "url": "https://datasets-server.huggingface.co/rows?dataset=LibrAI%2Fdo-not-answer&config=default&split=train&offset=0&length=100",
            "text_field": "question",
        },
    ]

    for source in sources:
        if len(results) >= max_examples:
            break
        try:
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            r = requests.get(source["url"], headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                rows = data.get("rows", [])
                for row in rows:
                    text = row.get("row", {}).get(source["text_field"], "")
                    if text and isinstance(text, str) and len(text.strip()) > 5:
                        results.append({
                            "text": text.strip(),
                            "intent_class_id": 2,
                            "intent_class_label": "adversarial_probe",
                            "tier": "1",
                            "language_iso": "en",
                            "resource_class": "hr_global",
                            "generated_by": f"hf_dataset:{source['url'].split('dataset=')[1].split('&')[0]}",
                            "split": "train",
                        })
                print(f"[ADVERSARIAL] Fetched {len(rows)} from {source['url'].split('dataset=')[1].split('&')[0]}")
            else:
                print(f"[ADVERSARIAL] HF fetch failed: HTTP {r.status_code}")
        except Exception as e:
            print(f"[ADVERSARIAL] Fetch error: {e}")

    print(f"[ADVERSARIAL] Total fetched: {len(results)}")
    return results[:max_examples]

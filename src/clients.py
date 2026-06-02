"""
API clients for all providers.
- Strict rate limiting: reads retry-after header precisely
- 2s sleep between Groq calls, 4s between Gemini calls
- Max 3 retries with exact header-guided sleep
- Designed to be called from parallel threads safely
"""

import os
import time
import json
import requests
from typing import Optional

GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

# ---------------------------------------------------------------------------
# Rate limit sleeps — strictly enforced
# ---------------------------------------------------------------------------

GROQ_INTER_REQUEST_SLEEP   = 2.1   # 30 RPM = 1 req/2s, add 0.1s buffer
GEMINI_INTER_REQUEST_SLEEP = 4.1   # 15 RPM = 1 req/4s, add 0.1s buffer
MISTRAL_INTER_REQUEST_SLEEP = 3.0  # conservative


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

def call_groq(prompt_system: str, prompt_user: str, model: str) -> Optional[str]:
    """Call Groq API. Reads retry-after header on 429. Returns content string or None."""
    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user",   "content": prompt_user},
        ],
        "temperature": 0.9,
        "max_tokens":  4096,
    }

    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)

            if r.status_code == 200:
                return _extract_openai_content(r.text, "groq")

            elif r.status_code == 429:
                retry_after = _get_retry_after(r)
                print(f"[groq] 429 — sleeping {retry_after}s (retry-after header)")
                time.sleep(retry_after)

            elif r.status_code == 400:
                data = r.json()
                err  = data.get("error", {}).get("message", "")
                if "decommissioned" in err or "not supported" in err:
                    print(f"[groq] Model decommissioned: {model}")
                    return None
                print(f"[groq] 400: {err[:200]}")
                return None

            elif r.status_code == 401:
                print(f"[groq] 401 Auth error — check GROQ_API_KEY")
                return None

            elif r.status_code in (500, 502, 503, 504):
                wait = (2 ** attempt) * 5
                print(f"[groq] Server error {r.status_code} — waiting {wait}s")
                time.sleep(wait)

            else:
                print(f"[groq] Unexpected {r.status_code}: {r.text[:150]}")
                time.sleep(5)

        except requests.exceptions.Timeout:
            print(f"[groq] Timeout attempt {attempt + 1}")
            time.sleep(10)
        except requests.exceptions.ConnectionError as e:
            print(f"[groq] Connection error: {e}")
            time.sleep(15)
        except Exception as e:
            print(f"[groq] Unexpected error: {e}")
            time.sleep(5)

    print(f"[groq] All 3 attempts failed for model {model}")
    return None


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def call_gemini(prompt_system: str, prompt_user: str, model: str) -> Optional[str]:
    """Call Gemini API via REST. Returns content string or None."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={GEMINI_API_KEY}")
    headers = {"Content-Type": "application/json"}

    # Gemini uses system_instruction separately
    payload = {
        "system_instruction": {
            "parts": [{"text": prompt_system}]
        },
        "contents": [
            {"parts": [{"text": prompt_user}]}
        ],
        "generationConfig": {
            "temperature":    0.9,
            "maxOutputTokens": 65536,  # Gemma 4 can output a lot
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)

            if r.status_code == 200:
                return _extract_gemini_content(r.text, model)

            elif r.status_code == 429:
                retry_after = _get_retry_after(r)
                print(f"[gemini/{model}] 429 — sleeping {retry_after}s")
                time.sleep(retry_after)

            elif r.status_code == 404:
                print(f"[gemini/{model}] 404 — model not found or not available")
                return None

            elif r.status_code == 400:
                print(f"[gemini/{model}] 400: {r.text[:200]}")
                return None

            elif r.status_code in (500, 502, 503, 504):
                wait = (2 ** attempt) * 5
                print(f"[gemini/{model}] {r.status_code} — waiting {wait}s")
                time.sleep(wait)

            else:
                print(f"[gemini/{model}] Unexpected {r.status_code}: {r.text[:150]}")
                time.sleep(5)

        except requests.exceptions.Timeout:
            print(f"[gemini/{model}] Timeout attempt {attempt + 1}")
            time.sleep(15)
        except Exception as e:
            print(f"[gemini/{model}] Error: {e}")
            time.sleep(5)

    print(f"[gemini/{model}] All 3 attempts failed")
    return None


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------

def call_mistral(prompt_system: str, prompt_user: str, model: str) -> Optional[str]:
    """Call Mistral API. Returns content string or None."""
    url     = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user",   "content": prompt_user},
        ],
        "temperature": 0.9,
        "max_tokens":  4096,
    }

    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)

            if r.status_code == 200:
                return _extract_openai_content(r.text, "mistral")

            elif r.status_code == 429:
                retry_after = _get_retry_after(r)
                print(f"[mistral] 429 — sleeping {retry_after}s")
                time.sleep(retry_after)

            elif r.status_code == 401:
                print(f"[mistral] 401 Auth error — check MISTRAL_API_KEY")
                return None

            elif r.status_code in (500, 502, 503, 504):
                wait = (2 ** attempt) * 5
                print(f"[mistral] {r.status_code} — waiting {wait}s")
                time.sleep(wait)

            else:
                print(f"[mistral] Unexpected {r.status_code}: {r.text[:150]}")
                time.sleep(5)

        except requests.exceptions.Timeout:
            print(f"[mistral] Timeout attempt {attempt + 1}")
            time.sleep(10)
        except Exception as e:
            print(f"[mistral] Error: {e}")
            time.sleep(5)

    print(f"[mistral] All 3 attempts failed")
    return None


# ---------------------------------------------------------------------------
# Adversarial — pull from existing HF red-team datasets
# ---------------------------------------------------------------------------

def fetch_adversarial_from_hf(max_examples: int = 300) -> list[dict]:
    """Pull from existing public adversarial datasets. Reformat to UEG schema."""
    from state import HF_TOKEN

    results = []
    sources = [
        {
            "url": "https://datasets-server.huggingface.co/rows?dataset=LibrAI%2Fdo-not-answer&config=default&split=train&offset=0&length=200",
            "text_field": "question",
            "source_name": "LibrAI/do-not-answer",
        },
        {
            "url": "https://datasets-server.huggingface.co/rows?dataset=walledai%2FAdvBench&config=default&split=train&offset=0&length=200",
            "text_field": "prompt",
            "source_name": "walledai/AdvBench",
        },
    ]

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    for source in sources:
        if len(results) >= max_examples:
            break
        try:
            r = requests.get(source["url"], headers=headers, timeout=30)
            if r.status_code == 200:
                rows = r.json().get("rows", [])
                for row in rows:
                    text = row.get("row", {}).get(source["text_field"], "")
                    if text and isinstance(text, str) and len(text.strip()) > 10:
                        # Filter out non-adversarial content from do-not-answer
                        # Keep only actual injection/jailbreak style prompts
                        text = text.strip()
                        results.append({
                            "text":               text,
                            "intent_class_id":    2,
                            "intent_class_label": "adversarial_probe",
                            "tier":               "1",
                            "language_iso":       "en",
                            "resource_class":     "hr_global",
                            "generated_by":       f"hf:{source['source_name']}",
                            "split":              "train",
                        })
                print(f"[ADV] Fetched {len(rows)} from {source['source_name']}")
            else:
                print(f"[ADV] {source['source_name']} failed: HTTP {r.status_code}")
        except Exception as e:
            print(f"[ADV] Fetch error: {e}")

    print(f"[ADV] Total adversarial examples fetched: {len(results)}")
    return results[:max_examples]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_retry_after(response) -> float:
    """Read retry-after header. Default to 30s if missing."""
    val = response.headers.get("retry-after", "30")
    try:
        return float(val) + 1.0  # add 1s buffer
    except (ValueError, TypeError):
        return 31.0


def _extract_openai_content(response_text: str, provider: str) -> Optional[str]:
    try:
        data = json.loads(response_text)
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[{provider}] Content extraction error: {e}")
        return None


def _extract_gemini_content(response_text: str, model: str) -> Optional[str]:
    try:
        data = json.loads(response_text)
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[gemini/{model}] Content extraction error: {e}")
        return None

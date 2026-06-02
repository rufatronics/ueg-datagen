"""
State management — single source of truth lives on HuggingFace.
Reads on startup, writes every 50 examples, writes on clean exit.
"""

import json
import os
import time
import requests
from datetime import datetime, timezone
from taxonomy import INTENT_CLASSES, TARGET_PER_CLASS

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = "rufatronics"
HF_DATASET_REPO = f"{HF_USERNAME}/ueg-training-data"
STATE_FILE = "progress.json"
HF_API = "https://huggingface.co/api"
HF_RAW = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"


def _hf_headers():
    return {"Authorization": f"Bearer {HF_TOKEN}"}


def load_state() -> dict:
    """Load progress state from HuggingFace. Returns fresh state if not found."""
    try:
        r = requests.get(f"{HF_RAW}/{STATE_FILE}", headers=_hf_headers(), timeout=30)
        if r.status_code == 200:
            state = r.json()
            print(f"[STATE] Loaded — {sum(state['class_counts'].values())} total examples so far")
            return state
        else:
            print(f"[STATE] No existing state found (HTTP {r.status_code}), starting fresh")
            return _fresh_state()
    except Exception as e:
        print(f"[STATE] Error loading state: {e} — starting fresh")
        return _fresh_state()


def _fresh_state() -> dict:
    return {
        "schema_version": "1.0",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "class_counts": {str(k): 0 for k in INTENT_CLASSES.keys()},
        "class_complete": {str(k): False for k in INTENT_CLASSES.keys()},
        "daily_usage": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "groq": {m: 0 for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                                      "deepseek-r1-distill-llama-70b", "qwen-qwq-32b"]},
            "gemini_flash_lite": 0,
            "gemini_flash": 0,
            "gemini_pro": 0,
            "mistral": 0,
            "openrouter": 0,
        },
        "total_generated": 0,
        "total_discarded": 0,
        "run_count": 0,
        "last_run_at": None,
        "completed_at": None,
    }


def save_state(state: dict) -> bool:
    """Push state JSON to HuggingFace dataset repo."""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Check if all classes done
    all_done = all(state["class_complete"].values())
    if all_done:
        state["status"] = "complete"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()

    content = json.dumps(state, indent=2)
    encoded = content.encode("utf-8")
    import base64
    b64 = base64.b64encode(encoded).decode("utf-8")

    # Get current SHA if file exists (needed for update)
    sha = _get_file_sha(STATE_FILE)

    payload = {
        "message": f"Update progress: {state['total_generated']} examples",
        "content": b64,
    }
    if sha:
        payload["sha"] = sha

    url = f"https://huggingface.co/api/datasets/{HF_DATASET_REPO}/upload/{STATE_FILE}"

    for attempt in range(3):
        try:
            r = requests.post(url, headers=_hf_headers(), json=payload, timeout=30)
            if r.status_code in (200, 201):
                print(f"[STATE] Saved — {state['total_generated']} total, {sum(1 for v in state['class_complete'].values() if v)} classes complete")
                return True
            else:
                print(f"[STATE] Save attempt {attempt+1} failed: HTTP {r.status_code} — {r.text[:200]}")
                time.sleep(5)
        except Exception as e:
            print(f"[STATE] Save attempt {attempt+1} exception: {e}")
            time.sleep(5)

    print("[STATE] CRITICAL: Failed to save state after 3 attempts")
    return False


def _get_file_sha(filename: str) -> str | None:
    """Get SHA of existing file on HF (needed for updates)."""
    try:
        url = f"https://huggingface.co/api/datasets/{HF_DATASET_REPO}/tree/main"
        r = requests.get(url, headers=_hf_headers(), timeout=15)
        if r.status_code == 200:
            for f in r.json():
                if f.get("path") == filename:
                    return f.get("oid")
    except Exception:
        pass
    return None


def reset_daily_usage_if_new_day(state: dict) -> dict:
    """Reset daily API counters if it's a new UTC day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["daily_usage"]["date"] != today:
        print(f"[STATE] New day {today} — resetting daily usage counters")
        state["daily_usage"] = {
            "date": today,
            "groq": {m: 0 for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                                      "deepseek-r1-distill-llama-70b", "qwen-qwq-32b"]},
            "gemini_flash_lite": 0,
            "gemini_flash": 0,
            "gemini_pro": 0,
            "mistral": 0,
            "openrouter": 0,
        }
    return state


def mark_class_complete(state: dict, class_id: int) -> dict:
    """Mark a class as done — no more writes to it."""
    state["class_complete"][str(class_id)] = True
    label = INTENT_CLASSES[class_id]["label"]
    count = state["class_counts"][str(class_id)]
    print(f"[STATE] ✓ Class {class_id} ({label}) COMPLETE — {count} examples")
    return state


def increment_class_count(state: dict, class_id: int, n: int) -> dict:
    state["class_counts"][str(class_id)] = state["class_counts"].get(str(class_id), 0) + n
    state["total_generated"] = state["total_generated"] + n
    count = state["class_counts"][str(class_id)]
    if count >= TARGET_PER_CLASS:
        state = mark_class_complete(state, class_id)
    return state


def get_remaining_for_class(state: dict, class_id: int) -> int:
    current = state["class_counts"].get(str(class_id), 0)
    return max(0, TARGET_PER_CLASS - current)


def is_class_done(state: dict, class_id: int) -> bool:
    return state["class_complete"].get(str(class_id), False)


def is_all_done(state: dict) -> bool:
    return state.get("status") == "complete" or all(state["class_complete"].values())


def print_progress(state: dict):
    print("\n" + "="*60)
    print(f"UEG DATA GENERATION PROGRESS")
    print(f"Total: {state['total_generated']} | Discarded: {state['total_discarded']}")
    print(f"Run #{state['run_count']} | Last: {state['last_updated']}")
    print("-"*60)
    for cid, info in INTENT_CLASSES.items():
        count = state["class_counts"].get(str(cid), 0)
        done = state["class_complete"].get(str(cid), False)
        pct = min(100, int(count / TARGET_PER_CLASS * 100))
        bar = ("█" * (pct // 5)).ljust(20)
        status = "✓" if done else " "
        print(f"  [{status}] {cid:2d} {info['label']:<28} {bar} {count:5d}/{TARGET_PER_CLASS}")
    print("="*60 + "\n")

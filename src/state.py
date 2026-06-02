"""
State management — single source of truth lives on HuggingFace.
Reads on startup, writes every 50 examples, writes on clean exit.
Uses new HuggingFace Hub API.
"""

import json
import os
import time
import tempfile
from datetime import datetime, timezone
from taxonomy import INTENT_CLASSES, TARGET_PER_CLASS

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = "rufatronics"
HF_DATASET_REPO = f"{HF_USERNAME}/ueg-training-data"
STATE_FILE = "progress.json"


def load_state() -> dict:
    """Load progress state from HuggingFace. Returns fresh state if not found."""
    try:
        from huggingface_hub import hf_hub_download
        
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=STATE_FILE,
            repo_type="dataset",
            token=HF_TOKEN,
            local_files_only=False
        )
        
        with open(local_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        total = sum(state['class_counts'].values())
        print(f"[STATE] Loaded — {total} total examples so far")
        return state
    except Exception as e:
        print(f"[STATE] No existing state found ({e}) — starting fresh")
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
            "groq": {m: 0 for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "meta-llama/llama-4-scout-17b-16e-instruct"]},
            "gemini_flash_lite": 0,
            "gemini_flash": 0,
            "gemini_pro": 0,
            "mistral": 0,
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

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=STATE_FILE,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"Update progress: {state['total_generated']} examples"
        )
        
        os.unlink(tmp_path)
        print(f"[STATE] Saved — {state['total_generated']} total, {sum(1 for v in state['class_complete'].values() if v)} classes complete")
        return True

    except Exception as e:
        print(f"[STATE] CRITICAL: Failed to save state — {e}")
        return False


def reset_daily_usage_if_new_day(state: dict) -> dict:
    """Reset daily API counters if it's a new UTC day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state["daily_usage"]["date"] != today:
        print(f"[STATE] New day {today} — resetting daily usage counters")
        state["daily_usage"] = {
            "date": today,
            "groq": {m: 0 for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "meta-llama/llama-4-scout-17b-16e-instruct"]},
            "gemini_flash_lite": 0,
            "gemini_flash": 0,
            "gemini_pro": 0,
            "mistral": 0,
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

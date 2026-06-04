"""
State management — HuggingFace is the single source of truth.
Reads on startup, checkpoints every 50 examples, saves on clean exit.
Recognizes existing progress perfectly — never double-counts, never overwrites.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from taxonomy import INTENT_CLASSES, TARGET_PER_CLASS, GROQ_MODELS, GROQ_RPD, GEMINI_MODELS, GEMINI_RPD

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_REPO     = "rufatronics/ueg-training-data"
STATE_FILE  = "progress.json"


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load state from HuggingFace. Returns fresh state if not found."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename=STATE_FILE,
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True,   # always get latest, not cached
        )
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        # Migrate old state schema if needed
        state = _migrate_state(state)

        total = state["total_generated"]
        done  = sum(1 for v in state["class_complete"].values() if v)
        print(f"[STATE] Loaded — {total:,} total examples, {done}/22 classes complete")
        return state

    except Exception as e:
        print(f"[STATE] No existing state ({e}) — starting fresh")
        return _fresh_state()


def save_state(state: dict) -> bool:
    """Push state to HuggingFace."""
    state["last_updated"] = _now()

    if all(state["class_complete"].values()):
        state["status"] = "complete"
        if not state.get("completed_at"):
            state["completed_at"] = _now()

    content = json.dumps(state, indent=2)

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)

        with tempfile.NamedTemporaryFile(mode="w", delete=False,
                                          suffix=".json", encoding="utf-8") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=STATE_FILE,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Progress: {state['total_generated']:,} examples | "
                           f"{sum(1 for v in state['class_complete'].values() if v)}/22 done",
        )
        os.unlink(tmp_path)

        total = state["total_generated"]
        done  = sum(1 for v in state["class_complete"].values() if v)
        print(f"[STATE] Saved — {total:,} total | {done}/22 classes complete")
        return True

    except Exception as e:
        print(f"[STATE] CRITICAL: Save failed — {e}")
        return False


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def reset_daily_if_new_day(state: dict) -> dict:
    today = _today()
    if state["daily_usage"]["date"] != today:
        print(f"[STATE] New day {today} — resetting daily counters")
        state["daily_usage"] = _fresh_daily()
    return state


def increment_count(state: dict, class_id: int, n: int) -> dict:
    sid = str(class_id)
    state["class_counts"][sid] = state["class_counts"].get(sid, 0) + n
    state["total_generated"] += n
    if state["class_counts"][sid] >= TARGET_PER_CLASS:
        if not state["class_complete"].get(sid, False):
            state["class_complete"][sid] = True
            label = INTENT_CLASSES[class_id]["label"]
            print(f"[STATE] ✓ Class {class_id} ({label}) COMPLETE — "
                  f"{state['class_counts'][sid]:,} examples")
    return state


def increment_discard(state: dict, n: int = 1) -> dict:
    state["total_discarded"] += n
    return state


def increment_api_usage(state: dict, provider: str, model: str) -> dict:
    usage = state["daily_usage"]
    if provider == "groq":
        usage["groq"][model] = usage["groq"].get(model, 0) + 1
    elif provider == "gemini":
        usage["gemini"][model] = usage["gemini"].get(model, 0) + 1
    elif provider == "mistral":
        if not isinstance(usage.get("mistral"), dict):
            usage["mistral"] = {}
        usage["mistral"][model] = usage["mistral"].get(model, 0) + 1
    return state


def is_class_done(state: dict, class_id: int) -> bool:
    return state["class_complete"].get(str(class_id), False)


def is_all_done(state: dict) -> bool:
    return state.get("status") == "complete" or all(
        state["class_complete"].get(str(cid), False)
        for cid in INTENT_CLASSES
    )


def remaining(state: dict, class_id: int) -> int:
    current = state["class_counts"].get(str(class_id), 0)
    return max(0, TARGET_PER_CLASS - current)


def groq_model_exhausted(state: dict, model: str) -> bool:
    used = state["daily_usage"]["groq"].get(model, 0)
    limit = GROQ_RPD.get(model, 1000)
    return used >= int(limit * 0.95)  # stop at 95% to leave headroom


def gemini_model_exhausted(state: dict, model: str) -> bool:
    used = state["daily_usage"]["gemini"].get(model, 0)
    limit = GEMINI_RPD.get(model, 20)
    return used >= int(limit * 0.95)


def print_progress(state: dict):
    print("\n" + "=" * 62)
    print(f"  UEG DATA GENERATION — RUN #{state['run_count']}")
    print(f"  Total: {state['total_generated']:,} | Discarded: {state['total_discarded']:,}")
    print(f"  Status: {state['status']} | Updated: {state['last_updated'][:19]}")
    print("-" * 62)
    for cid, info in INTENT_CLASSES.items():
        count = state["class_counts"].get(str(cid), 0)
        done  = state["class_complete"].get(str(cid), False)
        pct   = min(100, int(count / TARGET_PER_CLASS * 100))
        bar   = ("█" * (pct // 5)).ljust(20)
        tick  = "✓" if done else " "
        print(f"  [{tick}] {cid:2d} {info['label']:<28} {bar} {count:5,}/{TARGET_PER_CLASS:,}")
    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fresh_state() -> dict:
    return {
        "schema_version": "2.0",
        "status":         "running",
        "created_at":     _now(),
        "last_updated":   _now(),
        "completed_at":   None,
        "run_count":      0,
        "last_run_at":    None,
        "total_generated": 0,
        "total_discarded": 0,
        "class_counts":   {str(k): 0 for k in INTENT_CLASSES},
        "class_complete": {str(k): False for k in INTENT_CLASSES},
        "daily_usage":    _fresh_daily(),
    }


def _fresh_daily() -> dict:
    from taxonomy import GROQ_MODELS, GEMINI_MODELS, MISTRAL_MODELS
    return {
        "date":    _today(),
        "groq":    {m: 0 for m in GROQ_MODELS},
        "gemini":  {m: 0 for m in GEMINI_MODELS.values()},
        "mistral": {m: 0 for m in MISTRAL_MODELS.values()},
    }


def _migrate_state(state: dict) -> dict:
    """Forward-migrate old state schemas."""
    # Ensure all class keys exist
    for cid in INTENT_CLASSES:
        sid = str(cid)
        state["class_counts"].setdefault(sid, 0)
        state["class_complete"].setdefault(sid, False)
        # Re-check completion in case target was lowered/raised
        if state["class_counts"][sid] >= TARGET_PER_CLASS:
            state["class_complete"][sid] = True

    # Ensure daily_usage has current models
    state.setdefault("total_discarded", 0)
    state.setdefault("run_count", 0)
    state.setdefault("status", "running")
    du = state.setdefault("daily_usage", _fresh_daily())
    du.setdefault("groq", {m: 0 for m in GROQ_MODELS})
    du.setdefault("gemini", {m: 0 for m in GEMINI_MODELS.values()})
    # Migrate mistral from old int to new dict
    if not isinstance(du.get("mistral"), dict):
        du["mistral"] = {}
    from taxonomy import MISTRAL_MODELS
    for m in MISTRAL_MODELS.values():
        du["mistral"].setdefault(m, 0)
    # Add new Groq models if missing
    for m in GROQ_MODELS:
        du["groq"].setdefault(m, 0)
    for m in GEMINI_MODELS.values():
        du["gemini"].setdefault(m, 0)

    return state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

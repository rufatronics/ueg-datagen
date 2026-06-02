"""
Main generation engine.
Orchestrates provider routing, batch generation, verification, and pushing.
Checkpoints every 50 examples. Respects class completion flags.
"""

import time
import random
from typing import Optional

from taxonomy import (
    INTENT_CLASSES, GROQ_MODELS, GEMINI_MODELS, MISTRAL_MODELS,
    GROQ_CLASSES, MISTRAL_CLASSES, GEMINI_FLASH_LITE_CLASSES,
    GEMINI_FLASH_LANG_CLASSES, GEMINI_PRO_CLASSES,
    RESOURCE_CLASSES, TARGET_PER_CLASS,
)
from clients import call_groq, call_gemini, call_mistral, fetch_adversarial_from_hf
from prompts import build_prompt
from verify import verify_batch
from hf_push import push_examples
from state import (
    increment_class_count, is_class_done, get_remaining_for_class,
    save_state, is_all_done,
)

# Batch sizes per provider
BATCH_SIZES = {
    "groq":             15,
    "mistral":          10,
    "gemini_flash_lite": 20,
    "gemini_flash":     10,
    "gemini_pro":        5,
}

# Daily request limits (conservative — leave headroom)
DAILY_LIMITS = {
    "groq_per_model":    900,   # real limit ~1000, stop at 900
    "gemini_flash_lite": 900,
    "gemini_flash":      220,
    "gemini_pro":         90,
    "mistral":           400,   # token-paced, stop well before limit
}

# Languages for non-English generation via Gemini
LR_EMERGING_LANGS = [
    ("ha", "lr_emerging"), ("sw", "lr_emerging"), ("yo", "lr_emerging"),
    ("ig", "lr_emerging"), ("zu", "lr_emerging"), ("am", "lr_emerging"),
    ("so", "lr_emerging"), ("rw", "lr_emerging"),
]
MR_REGIONAL_LANGS = [
    ("ar", "mr_regional"), ("hi", "mr_regional"), ("zh", "mr_regional"),
    ("ja", "mr_regional"), ("ko", "mr_regional"), ("tr", "mr_regional"),
    ("vi", "mr_regional"), ("id", "mr_regional"),
]
MUL_MIX_LANGS = [
    ("pcm", "mul_mix"), ("hinglish", "mul_mix"),
    ("camfranglais", "mul_mix"), ("spanglish", "mul_mix"),
]
ALL_NON_ENGLISH = LR_EMERGING_LANGS + MR_REGIONAL_LANGS + MUL_MIX_LANGS


def _provider_exhausted(state: dict, provider: str, model: str = None) -> bool:
    """Check if a provider has hit its daily limit."""
    usage = state["daily_usage"]
    if provider == "groq" and model:
        return usage["groq"].get(model, 0) >= DAILY_LIMITS["groq_per_model"]
    elif provider == "gemini_flash_lite":
        return usage["gemini_flash_lite"] >= DAILY_LIMITS["gemini_flash_lite"]
    elif provider == "gemini_flash":
        return usage["gemini_flash"] >= DAILY_LIMITS["gemini_flash"]
    elif provider == "gemini_pro":
        return usage["gemini_pro"] >= DAILY_LIMITS["gemini_pro"]
    elif provider == "mistral":
        return usage["mistral"] >= DAILY_LIMITS["mistral"]
    return False


def _increment_usage(state: dict, provider: str, model: str = None) -> dict:
    if provider == "groq" and model:
        state["daily_usage"]["groq"][model] = state["daily_usage"]["groq"].get(model, 0) + 1
    elif provider in ("gemini_flash_lite", "gemini_flash", "gemini_pro", "mistral"):
        state["daily_usage"][provider] = state["daily_usage"].get(provider, 0) + 1
    return state


def _all_providers_exhausted(state: dict) -> bool:
    groq_done = all(
        _provider_exhausted(state, "groq", m) for m in GROQ_MODELS
    )
    gemini_done = (
        _provider_exhausted(state, "gemini_flash_lite") and
        _provider_exhausted(state, "gemini_flash") and
        _provider_exhausted(state, "gemini_pro")
    )
    mistral_done = _provider_exhausted(state, "mistral")
    return groq_done and gemini_done and mistral_done


def generate_for_class_groq(state: dict, class_id: int, model: str) -> tuple[dict, list[dict]]:
    """Generate a batch for a class using a specific Groq model."""
    if is_class_done(state, class_id) or _provider_exhausted(state, "groq", model):
        return state, []

    remaining = get_remaining_for_class(state, class_id)
    batch_size = min(BATCH_SIZES["groq"], remaining)
    if batch_size == 0:
        return state, []

    prompt = build_prompt(class_id, batch_size, "en", "hr_global")
    response = call_groq(prompt, model)
    state = _increment_usage(state, "groq", model)

    if response is None:
        print(f"[GEN] Groq/{model} class {class_id} — no response")
        state["total_discarded"] += 1
        return state, []

    valid, discarded = verify_batch(response, class_id)
    state["total_discarded"] += discarded

    for ex in valid:
        ex["generated_by"] = f"groq:{model}"

    print(f"[GEN] Groq/{model} class {class_id} — {len(valid)} valid, {discarded} discarded")
    return state, valid


def generate_for_class_mistral(state: dict, class_id: int) -> tuple[dict, list[dict]]:
    """Generate a batch using Mistral."""
    if is_class_done(state, class_id) or _provider_exhausted(state, "mistral"):
        return state, []

    remaining = get_remaining_for_class(state, class_id)
    batch_size = min(BATCH_SIZES["mistral"], remaining)
    if batch_size == 0:
        return state, []

    # Use small for Tier 4 and below, large for Tier 5
    tier = INTENT_CLASSES[class_id]["tier"]
    model = MISTRAL_MODELS["large"] if str(tier) in ("5A", "5B") else MISTRAL_MODELS["small"]

    prompt = build_prompt(class_id, batch_size, "en", "hr_global")
    response = call_mistral(prompt, model)
    state = _increment_usage(state, "mistral")

    if response is None:
        state["total_discarded"] += 1
        return state, []

    valid, discarded = verify_batch(response, class_id)
    state["total_discarded"] += discarded

    for ex in valid:
        ex["generated_by"] = f"mistral:{model}"

    print(f"[GEN] Mistral/{model} class {class_id} — {len(valid)} valid, {discarded} discarded")
    return state, valid


def generate_for_class_gemini(state: dict, class_id: int,
                               lang: str = "en", resource_class: str = "hr_global",
                               force_model: str = None) -> tuple[dict, list[dict]]:
    """Generate a batch using Gemini. Auto-selects model unless forced."""
    if is_class_done(state, class_id):
        return state, []

    # Pick model
    if force_model:
        model_key = force_model
    elif resource_class in ("lr_emerging", "mul_mix") and class_id in GEMINI_PRO_CLASSES:
        model_key = "gemini_pro"
    elif resource_class in ("lr_emerging", "mul_mix", "mr_regional"):
        model_key = "gemini_flash"
    else:
        model_key = "gemini_flash_lite"

    if _provider_exhausted(state, model_key):
        # Try fallback
        fallbacks = {"gemini_pro": "gemini_flash", "gemini_flash": "gemini_flash_lite"}
        fallback = fallbacks.get(model_key)
        if fallback and not _provider_exhausted(state, fallback):
            model_key = fallback
        else:
            return state, []

    remaining = get_remaining_for_class(state, class_id)
    batch_size = min(BATCH_SIZES[model_key], remaining)
    if batch_size == 0:
        return state, []

    model_name = GEMINI_MODELS[model_key.replace("gemini_", "").replace("_", "_")]
    # Fix key lookup
    key_map = {
        "gemini_flash_lite": "flash_lite",
        "gemini_flash": "flash",
        "gemini_pro": "pro",
    }
    model_name = GEMINI_MODELS[key_map[model_key]]

    prompt = build_prompt(class_id, batch_size, lang, resource_class)
    response = call_gemini(prompt, model_name)
    state = _increment_usage(state, model_key)

    if response is None:
        state["total_discarded"] += 1
        return state, []

    valid, discarded = verify_batch(response, class_id)
    state["total_discarded"] += discarded

    for ex in valid:
        ex["generated_by"] = f"gemini:{model_name}"
        ex["language_iso"] = lang
        ex["resource_class"] = resource_class

    print(f"[GEN] Gemini/{model_name} class {class_id} lang={lang} — {len(valid)} valid, {discarded} discarded")
    return state, valid


def run_adversarial_class(state: dict) -> tuple[dict, list[dict]]:
    """Pull adversarial examples from existing HF datasets."""
    if is_class_done(state, 2):
        return state, []

    remaining = get_remaining_for_class(state, 2)
    if remaining == 0:
        return state, []

    examples = fetch_adversarial_from_hf(max_examples=min(300, remaining))
    print(f"[GEN] Adversarial (HF datasets) — fetched {len(examples)} examples")
    return state, examples


def run_generation_cycle(state: dict, time_budget_seconds: int = 5400) -> dict:
    """
    Main loop. Runs until time budget exhausted, all providers exhausted,
    or all classes complete. Checkpoints to HF every 50 new examples.
    time_budget_seconds default = 90 minutes (leaves buffer for GH Actions 6hr limit).
    """
    start_time = time.time()
    pending: dict[int, list[dict]] = {cid: [] for cid in INTENT_CLASSES}
    examples_since_checkpoint = 0
    CHECKPOINT_EVERY = 50

    state["run_count"] = state.get("run_count", 0) + 1
    state["last_run_at"] = __import__("datetime").datetime.utcnow().isoformat()

    print(f"\n[ENGINE] Starting run #{state['run_count']}")
    print(f"[ENGINE] Time budget: {time_budget_seconds}s")

    if is_all_done(state):
        print("[ENGINE] All classes complete — nothing to do")
        return state

    # ---- Phase 1: Adversarial (class 02) from HF datasets ----
    if not is_class_done(state, 2):
        state, examples = run_adversarial_class(state)
        if examples:
            pending[2].extend(examples)
            state = increment_class_count(state, 2, len(examples))
            examples_since_checkpoint += len(examples)

    # ---- Phase 2: Groq — Tier 1-4 classes (except class 02) ----
    groq_classes = [c for c in GROQ_CLASSES if c != 2]
    groq_model_idx = 0

    for class_id in groq_classes:
        if time.time() - start_time > time_budget_seconds:
            print("[ENGINE] Time budget reached — stopping")
            break
        if is_class_done(state, class_id):
            continue

        # Round-robin across Groq models
        for _ in range(len(GROQ_MODELS)):
            model = GROQ_MODELS[groq_model_idx % len(GROQ_MODELS)]
            groq_model_idx += 1

            if not _provider_exhausted(state, "groq", model):
                state, examples = generate_for_class_groq(state, class_id, model)
                if examples:
                    pending[class_id].extend(examples)
                    state = increment_class_count(state, class_id, len(examples))
                    examples_since_checkpoint += len(examples)

                    if examples_since_checkpoint >= CHECKPOINT_EVERY:
                        state = _checkpoint(state, pending)
                        examples_since_checkpoint = 0
                        pending = {cid: [] for cid in INTENT_CLASSES}

                time.sleep(2)  # respect 30 RPM
                break

    # ---- Phase 3: Mistral — Tier 5A + 5B English ----
    for class_id in MISTRAL_CLASSES:
        if time.time() - start_time > time_budget_seconds:
            break
        if is_class_done(state, class_id):
            continue

        state, examples = generate_for_class_mistral(state, class_id)
        if examples:
            pending[class_id].extend(examples)
            state = increment_class_count(state, class_id, len(examples))
            examples_since_checkpoint += len(examples)

            if examples_since_checkpoint >= CHECKPOINT_EVERY:
                state = _checkpoint(state, pending)
                examples_since_checkpoint = 0
                pending = {cid: [] for cid in INTENT_CLASSES}

        time.sleep(3)

    # ---- Phase 4: Gemini Flash-Lite — Tier 5B English overflow ----
    for class_id in GEMINI_FLASH_LITE_CLASSES:
        if time.time() - start_time > time_budget_seconds:
            break
        if is_class_done(state, class_id):
            continue

        state, examples = generate_for_class_gemini(state, class_id, "en", "hr_global", "gemini_flash_lite")
        if examples:
            pending[class_id].extend(examples)
            state = increment_class_count(state, class_id, len(examples))
            examples_since_checkpoint += len(examples)

            if examples_since_checkpoint >= CHECKPOINT_EVERY:
                state = _checkpoint(state, pending)
                examples_since_checkpoint = 0
                pending = {cid: [] for cid in INTENT_CLASSES}

        time.sleep(4)

    # ---- Phase 5: Gemini — all non-English languages ----
    shuffled_langs = ALL_NON_ENGLISH.copy()
    random.shuffle(shuffled_langs)

    for lang, resource_class in shuffled_langs:
        if time.time() - start_time > time_budget_seconds:
            break

        # Pick a random class that isn't complete yet
        eligible = [c for c in range(1, 23) if not is_class_done(state, c)]
        if not eligible:
            break

        class_id = random.choice(eligible)
        state, examples = generate_for_class_gemini(state, class_id, lang, resource_class)
        if examples:
            pending[class_id].extend(examples)
            state = increment_class_count(state, class_id, len(examples))
            examples_since_checkpoint += len(examples)

            if examples_since_checkpoint >= CHECKPOINT_EVERY:
                state = _checkpoint(state, pending)
                examples_since_checkpoint = 0
                pending = {cid: [] for cid in INTENT_CLASSES}

        time.sleep(4)

    # ---- Final checkpoint ----
    has_pending = any(len(v) > 0 for v in pending.values())
    if has_pending or examples_since_checkpoint > 0:
        state = _checkpoint(state, pending)

    elapsed = int(time.time() - start_time)
    print(f"\n[ENGINE] Run complete — {elapsed}s elapsed, {state['total_generated']} total examples")

    return state


def _checkpoint(state: dict, pending: dict) -> dict:
    """Push all pending examples to HF and save state."""
    print(f"\n[CHECKPOINT] Pushing pending examples...")

    for class_id, examples in pending.items():
        if examples:
            success = push_examples(class_id, examples)
            if not success:
                print(f"[CHECKPOINT] WARNING: Failed to push class {class_id} — {len(examples)} examples lost")

    save_state(state)

    from hf_push import push_readme
    try:
        push_readme(state)
    except Exception as e:
        print(f"[CHECKPOINT] README update failed (non-critical): {e}")

    return state

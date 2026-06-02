"""
Main generation engine.
- Groq thread + Gemini thread run simultaneously
- Mistral runs sequentially after
- Strict per-provider sleeps enforced
- Checkpoints every 50 new examples
- Respects class completion — never writes to finished classes
"""

import time
import random
import threading
from datetime import datetime, timezone

from taxonomy import (
    INTENT_CLASSES, GROQ_MODELS, GEMINI_MODELS,
    MISTRAL_MODELS, BATCH_SIZES, TARGET_PER_CLASS,
    ENGLISH_ONLY_CLASSES,
)
from clients import (
    call_groq, call_gemini, call_mistral, fetch_adversarial_from_hf,
    GROQ_INTER_REQUEST_SLEEP, GEMINI_INTER_REQUEST_SLEEP, MISTRAL_INTER_REQUEST_SLEEP,
)
from prompts import build_single_class_prompt, build_mixed_class_prompt, SYSTEM_PROMPT
from verify import verify_batch
from hf_push import push_examples, push_readme
from state import (
    increment_count, increment_discard, increment_api_usage,
    is_class_done, is_all_done, remaining, save_state,
    groq_model_exhausted, gemini_model_exhausted,
)

CHECKPOINT_EVERY = 50   # push to HF every N new examples


# ---------------------------------------------------------------------------
# Shared checkpoint buffer — thread-safe
# ---------------------------------------------------------------------------

class CheckpointBuffer:
    def __init__(self, state: dict):
        self.state   = state
        self.pending: dict[int, list[dict]] = {cid: [] for cid in INTENT_CLASSES}
        self.count   = 0
        self.lock    = threading.Lock()

    def add(self, class_id: int, examples: list[dict], provider: str, model: str):
        if not examples:
            return
        with self.lock:
            for ex in examples:
                ex["generated_by"] = f"{provider}:{model}"
            self.pending[class_id].extend(examples)
            self.state = increment_count(self.state, class_id, len(examples))
            self.count += len(examples)

    def add_discard(self, n: int = 1):
        with self.lock:
            self.state = increment_discard(self.state, n)

    def flush_if_ready(self, force: bool = False) -> bool:
        with self.lock:
            if self.count < CHECKPOINT_EVERY and not force:
                return False
            return self._flush()

    def _flush(self) -> bool:
        has_data = any(len(v) > 0 for v in self.pending.values())
        if not has_data:
            return False

        print(f"\n[CHECKPOINT] Pushing {self.count} new examples...")
        for class_id, examples in self.pending.items():
            if examples:
                push_examples(class_id, examples)

        save_state(self.state)
        try:
            push_readme(self.state)
        except Exception as e:
            print(f"[CHECKPOINT] README failed (non-critical): {e}")

        self.pending = {cid: [] for cid in INTENT_CLASSES}
        self.count   = 0
        return True


# ---------------------------------------------------------------------------
# Groq thread
# ---------------------------------------------------------------------------

def _run_groq(buf: CheckpointBuffer, deadline: float):
    """Generate with all Groq models round-robin until time/quota exhausted."""
    model_idx = 0
    all_class_ids = [cid for cid in INTENT_CLASSES if cid != 2]  # skip adversarial

    while time.time() < deadline:
        if is_all_done(buf.state):
            break

        model = GROQ_MODELS[model_idx % len(GROQ_MODELS)]
        model_idx += 1

        if groq_model_exhausted(buf.state, model):
            # Check if ALL Groq models are exhausted
            if all(groq_model_exhausted(buf.state, m) for m in GROQ_MODELS):
                print("[Groq] All models exhausted for today")
                break
            continue

        # Pick a class that still needs work
        eligible = [cid for cid in all_class_ids if not is_class_done(buf.state, cid)]
        if not eligible:
            break

        class_id = random.choice(eligible)
        rem      = remaining(buf.state, class_id)
        if rem == 0:
            continue

        batch_size = min(BATCH_SIZES["groq"], rem)
        sys_p, usr_p, lang = build_single_class_prompt(class_id, batch_size)

        response = call_groq(sys_p, usr_p, model)

        with buf.lock:
            buf.state = increment_api_usage(buf.state, "groq", model)

        time.sleep(GROQ_INTER_REQUEST_SLEEP)

        if response is None:
            buf.add_discard()
            continue

        valid, discards = verify_batch(response, expected_class_id=class_id)
        buf.add_discard(discards)

        if valid:
            buf.add(class_id, valid, "groq", model)
            print(f"[Groq/{model.split('/')[-1][:20]}] class {class_id} "
                  f"lang={lang} — {len(valid)} valid, {discards} discarded")
            buf.flush_if_ready()


# ---------------------------------------------------------------------------
# Gemini thread
# ---------------------------------------------------------------------------

def _run_gemini(buf: CheckpointBuffer, deadline: float):
    """Generate with Gemma 4 (mixed-class, large batches) + Flash models (single-class)."""

    # Gemma 4 models — big mixed batches
    gemma_models = [GEMINI_MODELS["gemma4_26b"], GEMINI_MODELS["gemma4_31b"]]
    flash_models = [GEMINI_MODELS["flash_lite"], GEMINI_MODELS["flash"]]

    gemma_idx = 0
    flash_idx = 0

    all_class_ids = list(INTENT_CLASSES.keys())

    while time.time() < deadline:
        if is_all_done(buf.state):
            break

        eligible = [cid for cid in all_class_ids if not is_class_done(buf.state, cid)]
        if not eligible:
            break

        # --- Gemma 4 mixed-class batch ---
        model = gemma_models[gemma_idx % len(gemma_models)]
        gemma_idx += 1

        if not gemini_model_exhausted(buf.state, model):
            # Pick 4-6 random eligible classes for the mixed batch
            mix_size   = min(6, len(eligible))
            mix_classes = random.sample(eligible, mix_size)
            total       = BATCH_SIZES["gemma4"]  # 80 examples total across classes

            sys_p, usr_p = build_mixed_class_prompt(mix_classes, total)
            response     = call_gemini(sys_p, usr_p, model)

            with buf.lock:
                buf.state = increment_api_usage(buf.state, "gemini", model)

            time.sleep(GEMINI_INTER_REQUEST_SLEEP)

            if response is not None:
                # Mixed-class mode — no expected_class_id
                valid, discards = verify_batch(response, expected_class_id=None)
                buf.add_discard(discards)

                # Group by class
                by_class: dict[int, list[dict]] = {}
                for ex in valid:
                    cid = ex["intent_class_id"]
                    by_class.setdefault(cid, []).append(ex)

                for cid, examples in by_class.items():
                    if not is_class_done(buf.state, cid):
                        buf.add(cid, examples, "gemini", model)

                total_valid = sum(len(v) for v in by_class.values())
                print(f"[Gemini/{model}] mixed {mix_classes} — "
                      f"{total_valid} valid, {discards} discarded")
                buf.flush_if_ready()
            else:
                buf.add_discard()

        # --- Gemini Flash single-class ---
        flash_model = flash_models[flash_idx % len(flash_models)]
        flash_idx += 1

        if not gemini_model_exhausted(buf.state, flash_model):
            # Pick one class — prioritize non-English eligible
            class_id = random.choice(eligible)
            rem      = remaining(buf.state, class_id)
            if rem > 0:
                batch_size = min(BATCH_SIZES["gemini_flash"], rem)
                sys_p, usr_p, lang = build_single_class_prompt(class_id, batch_size)
                response = call_gemini(sys_p, usr_p, flash_model)

                with buf.lock:
                    buf.state = increment_api_usage(buf.state, "gemini", flash_model)

                time.sleep(GEMINI_INTER_REQUEST_SLEEP)

                if response is not None:
                    valid, discards = verify_batch(response, expected_class_id=class_id)
                    buf.add_discard(discards)
                    if valid:
                        buf.add(class_id, valid, "gemini", flash_model)
                        print(f"[Gemini/{flash_model}] class {class_id} "
                              f"lang={lang} — {len(valid)} valid, {discards} discarded")
                        buf.flush_if_ready()
                else:
                    buf.add_discard()


# ---------------------------------------------------------------------------
# Mistral (sequential)
# ---------------------------------------------------------------------------

def _run_mistral(buf: CheckpointBuffer, deadline: float):
    """Generate with Mistral sequentially."""
    all_class_ids = [cid for cid in INTENT_CLASSES if cid != 2]

    while time.time() < deadline:
        if is_all_done(buf.state):
            break

        eligible = [cid for cid in all_class_ids if not is_class_done(buf.state, cid)]
        if not eligible:
            break

        class_id = random.choice(eligible)
        rem      = remaining(buf.state, class_id)
        if rem == 0:
            continue

        # Use large for Tier 5, small for everything else
        tier  = INTENT_CLASSES[class_id]["tier"]
        model = MISTRAL_MODELS["large"] if str(tier) in ("5A", "5B") else MISTRAL_MODELS["small"]

        batch_size = min(BATCH_SIZES["mistral"], rem)
        sys_p, usr_p, lang = build_single_class_prompt(class_id, batch_size)

        response = call_mistral(sys_p, usr_p, model)

        with buf.lock:
            buf.state = increment_api_usage(buf.state, "mistral", model)

        time.sleep(MISTRAL_INTER_REQUEST_SLEEP)

        if response is None:
            buf.add_discard()
            continue

        valid, discards = verify_batch(response, expected_class_id=class_id)
        buf.add_discard(discards)

        if valid:
            buf.add(class_id, valid, "mistral", model)
            print(f"[Mistral/{model}] class {class_id} lang={lang} — "
                  f"{len(valid)} valid, {discards} discarded")
            buf.flush_if_ready()


# ---------------------------------------------------------------------------
# Adversarial (class 02) — separate, from HF datasets
# ---------------------------------------------------------------------------

def _run_adversarial(buf: CheckpointBuffer):
    """Top up adversarial class from HF red-team datasets if not done."""
    if is_class_done(buf.state, 2):
        return

    rem      = remaining(buf.state, 2)
    examples = fetch_adversarial_from_hf(max_examples=min(300, rem))
    if examples:
        buf.add(2, examples, "hf_dataset", "do-not-answer+advbench")
        print(f"[ADV] Added {len(examples)} adversarial examples")
        buf.flush_if_ready(force=True)


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_generation_cycle(state: dict, time_budget_seconds: int = 5400) -> dict:
    """
    Run one generation cycle.
    Groq + Gemini run in parallel threads.
    Mistral runs after.
    Adversarial class topped up at start.
    """
    start    = time.time()
    deadline = start + time_budget_seconds

    state["run_count"] = state.get("run_count", 0) + 1
    state["last_run_at"] = datetime.now(timezone.utc).isoformat()

    print(f"\n[ENGINE] Starting run #{state['run_count']}")
    print(f"[ENGINE] Budget: {time_budget_seconds // 60} minutes")
    print(f"[ENGINE] Target: 176,000 examples across 22 classes\n")

    if is_all_done(state):
        print("[ENGINE] All classes complete — nothing to do")
        return state

    buf = CheckpointBuffer(state)

    # Phase 1 — adversarial class (quick, from HF)
    _run_adversarial(buf)

    # Phase 2 — Groq + Gemini in parallel
    groq_thread   = threading.Thread(target=_run_groq,   args=(buf, deadline), daemon=True)
    gemini_thread = threading.Thread(target=_run_gemini, args=(buf, deadline), daemon=True)

    groq_thread.start()
    gemini_thread.start()

    groq_thread.join()
    gemini_thread.join()

    # Phase 3 — Mistral (remaining budget)
    remaining_budget = int(deadline - time.time())
    if remaining_budget > 60:
        print(f"\n[ENGINE] Starting Mistral phase ({remaining_budget}s remaining)")
        _run_mistral(buf, deadline)

    # Final checkpoint
    buf.flush_if_ready(force=True)

    elapsed = int(time.time() - start)
    print(f"\n[ENGINE] Run complete — {elapsed}s elapsed")
    print(f"[ENGINE] Generated this run: {sum(len(v) for v in buf.pending.values()) + (buf.count)} examples")

    return buf.state

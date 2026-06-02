"""
Main entrypoint — called by GitHub Actions.
Loads state, runs generation cycle, saves state, exits cleanly.
"""

import sys
import signal
import traceback
from datetime import datetime, timezone

from state import load_state, save_state, reset_daily_usage_if_new_day, print_progress, is_all_done
from generator import run_generation_cycle
from hf_push import ensure_repo_exists

# Graceful shutdown on SIGTERM (GitHub Actions sends this before killing)
_state_ref = [None]

def _handle_sigterm(signum, frame):
    print("\n[MAIN] SIGTERM received — saving state before exit")
    if _state_ref[0] is not None:
        save_state(_state_ref[0])
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)


def main():
    print(f"\n{'='*60}")
    print(f"UEG DATA GENERATOR")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # Step 1 — ensure HF repo exists
    ensure_repo_exists()

    # Step 2 — load state
    state = load_state()
    _state_ref[0] = state

    # Step 3 — reset daily counters if new day
    state = reset_daily_usage_if_new_day(state)

    # Step 4 — print current progress
    print_progress(state)

    # Step 5 — check if already done
    if is_all_done(state):
        print("[MAIN] ✅ All classes complete — dataset generation finished!")
        print(f"[MAIN] Total examples: {state['total_generated']}")
        sys.exit(0)

    # Step 6 — run generation
    try:
        state = run_generation_cycle(state, time_budget_seconds=5400)  # 90 min
        _state_ref[0] = state
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted — saving state")
        save_state(state)
        sys.exit(0)
    except Exception as e:
        print(f"\n[MAIN] FATAL ERROR: {e}")
        traceback.print_exc()
        print("[MAIN] Saving state before crash exit")
        save_state(state)
        sys.exit(1)

    # Step 7 — final state save + progress report
    save_state(state)
    print_progress(state)

    if is_all_done(state):
        print("\n[MAIN] ✅ COMPLETE — all 22 classes have reached their targets!")
        print(f"[MAIN] Total examples generated: {state['total_generated']}")
        print(f"[MAIN] Total discarded: {state['total_discarded']}")
    else:
        done = sum(1 for v in state["class_complete"].values() if v)
        print(f"\n[MAIN] Run finished — {done}/22 classes complete")
        print(f"[MAIN] Total so far: {state['total_generated']}")
        print("[MAIN] Next run scheduled via GitHub Actions cron")

    sys.exit(0)


if __name__ == "__main__":
    main()

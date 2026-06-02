"""
Main entrypoint — called by GitHub Actions every 4 hours.
Loads state from HF, runs generation, saves state, exits cleanly.
"""

import sys
import signal
import traceback
from datetime import datetime, timezone

from state import load_state, save_state, reset_daily_if_new_day, print_progress, is_all_done
from generator import run_generation_cycle
from hf_push import ensure_repo_exists

# Graceful SIGTERM handler (GitHub Actions sends this before killing)
_state_ref = [None]

def _handle_sigterm(signum, frame):
    print("\n[MAIN] SIGTERM — saving state before exit")
    if _state_ref[0] is not None:
        save_state(_state_ref[0])
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)


def main():
    print(f"\n{'=' * 62}")
    print(f"  UEG DATA GENERATOR v2")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'=' * 62}\n")

    # Ensure HF repo exists
    ensure_repo_exists()

    # Load state — recognizes existing progress perfectly
    state = load_state()
    _state_ref[0] = state

    # Reset daily counters if new UTC day
    state = reset_daily_if_new_day(state)

    # Show current progress
    print_progress(state)

    # Already done?
    if is_all_done(state):
        print("[MAIN] ✅ All 22 classes complete. Dataset generation finished!")
        print(f"[MAIN] Total: {state['total_generated']:,} examples")
        sys.exit(0)

    # Check for STOP flag
    import os
    if os.path.exists("STOP"):
        print("[MAIN] STOP file found — skipping run")
        sys.exit(0)

    # Run generation
    try:
        state = run_generation_cycle(state, time_budget_seconds=5400)
        _state_ref[0] = state
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted — saving state")
        save_state(state)
        sys.exit(0)
    except Exception as e:
        print(f"\n[MAIN] FATAL: {e}")
        traceback.print_exc()
        print("[MAIN] Saving state before exit")
        save_state(state)
        sys.exit(1)

    # Final save + progress report
    save_state(state)
    print_progress(state)

    if is_all_done(state):
        print("\n[MAIN] ✅ COMPLETE — all 22 classes done!")
        print(f"[MAIN] Total: {state['total_generated']:,} | Discarded: {state['total_discarded']:,}")
    else:
        done = sum(1 for v in state["class_complete"].values() if v)
        print(f"\n[MAIN] Run done — {done}/22 classes complete, {state['total_generated']:,} total")
        print("[MAIN] Next run in ~4 hours via GitHub Actions cron")

    sys.exit(0)


if __name__ == "__main__":
    main()

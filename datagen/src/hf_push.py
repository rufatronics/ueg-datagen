"""
HuggingFace dataset pusher.
One JSONL file per class. Appends only — never overwrites.
Uses upload_folder — ALL changed files in ONE commit.
Eliminates the 128 commits/hour rate limit problem entirely.
"""

import os
import json
import time
import shutil
import tempfile
from taxonomy import INTENT_CLASSES, TARGET_PER_CLASS

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO  = "rufatronics/ueg-training-data"


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)


def _class_filename(class_id: int) -> str:
    label = INTENT_CLASSES[class_id]["label"]
    return f"class_{class_id:02d}_{label}.jsonl"


def push_batch(pending: dict[int, list[dict]], state: dict) -> bool:
    """
    Push ALL pending examples across ALL classes in ONE commit using upload_folder.
    pending = {class_id: [list of new examples]}
    This is the core fix — 1 commit regardless of how many classes changed.
    """
    if not any(len(v) > 0 for v in pending.values()):
        return True

    api      = _api()
    tmp_dir  = tempfile.mkdtemp()
    data_dir = os.path.join(tmp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    total_new = 0

    try:
        for class_id, new_examples in pending.items():
            if not new_examples:
                continue

            filename   = _class_filename(class_id)
            local_path = os.path.join(data_dir, filename)

            # Fetch existing content from HF
            existing = ""
            try:
                from huggingface_hub import hf_hub_download
                dl = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=f"data/{filename}",
                    repo_type="dataset",
                    token=HF_TOKEN,
                    force_download=True,
                )
                with open(dl, "r", encoding="utf-8") as f:
                    existing = f.read()
            except Exception:
                existing = ""  # file doesn't exist yet — fine

            # Append new examples
            new_lines = "\n".join(
                json.dumps(ex, ensure_ascii=False) for ex in new_examples
            ) + "\n"

            with open(local_path, "w", encoding="utf-8") as f:
                f.write(existing + new_lines)

            total_new += len(new_examples)
            print(f"[HF] Staged {len(new_examples):,} examples → data/{filename}")

        # Also write updated progress.json into the folder
        progress_path = os.path.join(tmp_dir, "progress.json")
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Write updated README
        readme_path = os.path.join(tmp_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(_build_readme(state))

        # ONE upload_folder call = ONE commit — solves the rate limit
        for attempt in range(3):
            try:
                api.upload_folder(
                    folder_path=tmp_dir,
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    commit_message=(
                        f"Checkpoint: +{total_new:,} examples | "
                        f"{state['total_generated']:,} total | "
                        f"{sum(1 for v in state['class_complete'].values() if v)}/22 done"
                    ),
                    ignore_patterns=["*.pyc", "__pycache__"],
                )
                print(f"[HF] ✓ Pushed {total_new:,} new examples in 1 commit")
                return True

            except Exception as e:
                err = str(e)
                if "rate limit" in err.lower() or "128" in err:
                    # HF commit rate limit — wait it out
                    print(f"[HF] Commit rate limit hit — waiting 15 minutes")
                    time.sleep(900)
                else:
                    print(f"[HF] Upload attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(15 * (attempt + 1))

        print("[HF] FAILED to push after 3 attempts")
        return False

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def ensure_repo_exists():
    """Create HF dataset repo if it doesn't exist."""
    try:
        api = _api()
        try:
            api.dataset_info(repo_id=HF_REPO, token=HF_TOKEN)
            print("[HF] Dataset repo exists")
        except Exception:
            api.create_repo(
                repo_id=HF_REPO,
                repo_type="dataset",
                private=False,
                token=HF_TOKEN,
            )
            print("[HF] Dataset repo created")
    except Exception as e:
        print(f"[HF] Repo check error: {e}")


def _build_readme(state: dict) -> str:
    lines = [
        "# UEG Training Data\n\n",
        "Auto-generated training dataset for the Universal Edge Gateway (UEG) "
        "intent classifier.\n\n",
        f"**Total examples:** {state['total_generated']:,}  \n",
        f"**Status:** {state['status']}  \n",
        f"**Last updated:** {state['last_updated'][:19]} UTC  \n\n",
        "## Progress\n\n",
        "| # | Class | Count | Target | Done |\n",
        "|---|-------|-------|--------|------|\n",
    ]
    for cid, info in INTENT_CLASSES.items():
        count = state["class_counts"].get(str(cid), 0)
        done  = "✅" if state["class_complete"].get(str(cid), False) else "🔄"
        lines.append(
            f"| {cid} | {info['label']} | {count:,} | {TARGET_PER_CLASS:,} | {done} |\n"
        )
    return "".join(lines)


# Keep push_readme as a standalone for state.py compatibility
def push_readme(state: dict):
    """Standalone README push — used by state save."""
    pass  # README is now bundled into push_batch — no separate commit needed

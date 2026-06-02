"""
HuggingFace dataset pusher.
One JSONL file per class: data/class_01_noise_gibberish.jsonl etc.
Appends to existing files — never overwrites.
Uses new /commit/ endpoint (old /upload/ is deprecated).
"""

import os
import json
import time
import base64
import requests
from typing import Optional
from huggingface_hub import CommitScheduler, CommitOperationAdd
import tempfile

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = "rufatronics"
HF_REPO     = f"{HF_USERNAME}/ueg-training-data"

from taxonomy import INTENT_CLASSES


def _class_filename(class_id: int) -> str:
    label = INTENT_CLASSES[class_id]["label"]
    return f"data/class_{class_id:02d}_{label}.jsonl"


def push_examples(class_id: int, examples: list[dict]) -> bool:
    """
    Append new examples to the class JSONL file on HuggingFace.
    Uses huggingface_hub library for reliable uploads.
    Creates the file if it doesn't exist.
    Returns True on success.
    """
    if not examples:
        return True

    path = _class_filename(class_id)

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        
        # Build new lines
        new_content = ""
        for ex in examples:
            new_content += json.dumps(ex, ensure_ascii=False) + "\n"

        # Try to fetch existing content
        try:
            existing_content = api.hf_hub_download(
                repo_id=HF_REPO,
                filename=path,
                repo_type="dataset",
                token=HF_TOKEN
            )
            with open(existing_content, 'r', encoding='utf-8') as f:
                full_content = f.read() + new_content
        except Exception:
            # File doesn't exist yet
            full_content = new_content

        # Create temp file with combined content
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl', encoding='utf-8') as tmp:
            tmp.write(full_content)
            tmp_path = tmp.name

        # Upload using new API
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=path,
            repo_id=HF_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"Add {len(examples)} examples to class {class_id}"
        )

        os.unlink(tmp_path)
        print(f"[HF] ✓ Pushed {len(examples)} examples → {path}")
        return True

    except Exception as e:
        print(f"[HF] Push failed for class {class_id}: {e}")
        return False


def push_bulk(batches: dict[int, list[dict]]) -> dict[int, bool]:
    """Push multiple classes at once. Returns {class_id: success}."""
    results = {}
    for class_id, examples in batches.items():
        if examples:
            results[class_id] = push_examples(class_id, examples)
            time.sleep(1)  # small pause between pushes
    return results


def ensure_repo_exists():
    """Create the HF dataset repo if it doesn't exist yet."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        try:
            api.dataset_info(repo_id=HF_REPO, token=HF_TOKEN)
            print("[HF] Dataset repo already exists")
        except Exception:
            api.create_repo(
                repo_id=HF_REPO,
                repo_type="dataset",
                private=False,
                token=HF_TOKEN
            )
            print("[HF] Dataset repo created")
    except Exception as e:
        print(f"[HF] Repo check error: {e}")


def push_readme(state: dict):
    """Push a README with current progress stats."""
    from taxonomy import INTENT_CLASSES, TARGET_PER_CLASS

    lines = [
        "# UEG Training Data\n",
        "Auto-generated training dataset for the Universal Edge Gateway (UEG) intent classifier.\n",
        f"**Total examples:** {state['total_generated']:,}\n",
        f"**Status:** {state['status']}\n",
        f"**Last updated:** {state['last_updated']}\n\n",
        "## Progress per class\n\n",
        "| Class | Label | Count | Target | Done |\n",
        "|-------|-------|-------|--------|------|\n",
    ]
    for cid, info in INTENT_CLASSES.items():
        count = state["class_counts"].get(str(cid), 0)
        done = "✅" if state["class_complete"].get(str(cid), False) else "🔄"
        lines.append(f"| {cid} | {info['label']} | {count:,} | {TARGET_PER_CLASS:,} | {done} |\n")

    content = "".join(lines)

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md', encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=HF_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message="Update README progress"
        )
        
        os.unlink(tmp_path)
        print("[HF] README updated")
    except Exception as e:
        print(f"[HF] README update error: {e}")

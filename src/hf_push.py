"""
HuggingFace dataset pusher.
One JSONL file per class: data/class_01_noise_gibberish.jsonl etc.
Appends to existing files — never overwrites.
"""

import os
import json
import time
import base64
import requests
from typing import Optional

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = "rufatronics"
HF_REPO     = f"{HF_USERNAME}/ueg-training-data"
HF_API_BASE = f"https://huggingface.co/api/datasets/{HF_REPO}"

from taxonomy import INTENT_CLASSES


def _headers():
    return {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}


def _class_filename(class_id: int) -> str:
    label = INTENT_CLASSES[class_id]["label"]
    return f"data/class_{class_id:02d}_{label}.jsonl"


def _get_file_info(path: str) -> tuple[Optional[str], Optional[str]]:
    """Returns (current_content_b64, sha) or (None, None) if file doesn't exist."""
    try:
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{path}"
        r = requests.get(url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=30)
        if r.status_code == 200:
            existing = r.text

            # Get SHA for update
            tree_url = f"{HF_API_BASE}/tree/main/data"
            tr = requests.get(tree_url, headers=_headers(), timeout=15)
            sha = None
            if tr.status_code == 200:
                for f in tr.json():
                    if f.get("path") == path:
                        sha = f.get("oid")
                        break
            return existing, sha
        return None, None
    except Exception as e:
        print(f"[HF] Error getting file info for {path}: {e}")
        return None, None


def push_examples(class_id: int, examples: list[dict]) -> bool:
    """
    Append new examples to the class JSONL file on HuggingFace.
    Creates the file if it doesn't exist.
    Returns True on success.
    """
    if not examples:
        return True

    path = _class_filename(class_id)

    # Build new lines
    new_lines = ""
    for ex in examples:
        new_lines += json.dumps(ex, ensure_ascii=False) + "\n"

    # Fetch existing content
    existing_content, sha = _get_file_info(path)

    if existing_content is not None:
        full_content = existing_content + new_lines
    else:
        full_content = new_lines

    # Encode and push
    encoded = base64.b64encode(full_content.encode("utf-8")).decode("utf-8")

    payload = {
        "message": f"Add {len(examples)} examples to class {class_id}",
        "content": encoded,
    }
    if sha:
        payload["sha"] = sha

    url = f"{HF_API_BASE}/upload/{path}"

    for attempt in range(3):
        try:
            r = requests.post(url, headers=_headers(), json=payload, timeout=60)
            if r.status_code in (200, 201):
                print(f"[HF] ✓ Pushed {len(examples)} examples → {path}")
                return True
            elif r.status_code == 429:
                wait = (2 ** attempt) * 15
                print(f"[HF] Rate limited — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"[HF] Push failed attempt {attempt+1}: HTTP {r.status_code} — {r.text[:200]}")
                time.sleep(10)
        except Exception as e:
            print(f"[HF] Push exception attempt {attempt+1}: {e}")
            time.sleep(10)

    print(f"[HF] FAILED to push class {class_id} after 3 attempts")
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
    url = f"https://huggingface.co/api/repos/create"
    payload = {
        "type": "dataset",
        "name": "ueg-training-data",
        "private": False,
    }
    try:
        r = requests.post(url, headers=_headers(), json=payload, timeout=30)
        if r.status_code in (200, 201):
            print("[HF] Dataset repo created")
        elif r.status_code == 409:
            print("[HF] Dataset repo already exists")
        else:
            print(f"[HF] Repo creation: HTTP {r.status_code} — {r.text[:200]}")
    except Exception as e:
        print(f"[HF] Repo creation error: {e}")


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
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    sha = None
    try:
        tree_url = f"{HF_API_BASE}/tree/main"
        tr = requests.get(tree_url, headers=_headers(), timeout=15)
        if tr.status_code == 200:
            for f in tr.json():
                if f.get("path") == "README.md":
                    sha = f.get("oid")
    except Exception:
        pass

    payload = {"message": "Update README progress", "content": encoded}
    if sha:
        payload["sha"] = sha

    try:
        r = requests.post(f"{HF_API_BASE}/upload/README.md",
                          headers=_headers(), json=payload, timeout=30)
        if r.status_code in (200, 201):
            print("[HF] README updated")
    except Exception as e:
        print(f"[HF] README update error: {e}")

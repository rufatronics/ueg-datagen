"""
HuggingFace dataset pusher.
One JSONL file per class. Appends only — never overwrites.
Uses huggingface_hub library (new commit API).
"""

import os
import json
import time
import tempfile
from taxonomy import INTENT_CLASSES, TARGET_PER_CLASS

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO  = "rufatronics/ueg-training-data"


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)


def _class_path(class_id: int) -> str:
    label = INTENT_CLASSES[class_id]["label"]
    return f"data/class_{class_id:02d}_{label}.jsonl"


def push_examples(class_id: int, examples: list[dict]) -> bool:
    """Append examples to the class JSONL file. Creates if absent."""
    if not examples:
        return True

    path = _class_path(class_id)
    api  = _api()

    # Fetch existing content
    try:
        local = api.hf_hub_download(
            repo_id=HF_REPO, filename=path,
            repo_type="dataset", token=HF_TOKEN,
            force_download=True,
        )
        with open(local, "r", encoding="utf-8") as f:
            existing = f.read()
    except Exception:
        existing = ""

    # Build new content
    new_lines = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples) + "\n"
    full      = existing + new_lines

    # Write and upload
    for attempt in range(3):
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False,
                                             suffix=".jsonl", encoding="utf-8") as tmp:
                tmp.write(full)
                tmp_path = tmp.name

            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=path,
                repo_id=HF_REPO,
                repo_type="dataset",
                commit_message=f"Add {len(examples)} examples → class {class_id}",
            )
            os.unlink(tmp_path)
            print(f"[HF] ✓ {len(examples)} examples → {path}")
            return True

        except Exception as e:
            print(f"[HF] Push attempt {attempt + 1} failed for class {class_id}: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            if attempt < 2:
                time.sleep(10 * (attempt + 1))

    print(f"[HF] FAILED to push class {class_id} after 3 attempts")
    return False


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


def push_readme(state: dict):
    """Update README with current progress."""
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
        lines.append(f"| {cid} | {info['label']} | {count:,} | {TARGET_PER_CLASS:,} | {done} |\n")

    content = "".join(lines)

    try:
        api = _api()
        with tempfile.NamedTemporaryFile(mode="w", delete=False,
                                          suffix=".md", encoding="utf-8") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message="Update README",
        )
        os.unlink(tmp_path)
        print("[HF] README updated")
    except Exception as e:
        print(f"[HF] README update failed (non-critical): {e}")

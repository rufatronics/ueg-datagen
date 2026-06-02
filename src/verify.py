"""
Verification — JSON-only gate.
Valid JSON + correct fields + class matches request = passes.
Everything else is discarded.
"""

import json
from taxonomy import INTENT_CLASSES, VALID_CLASS_IDS, VALID_RESOURCE_CLASSES


def verify_batch(raw_response: str, expected_class_id: int) -> tuple[list[dict], int]:
    """
    Parse and verify a raw model response.
    Returns (valid_examples, discard_count).
    """
    valid = []
    discarded = 0

    # Step 1 — strip any markdown fences models sneak in
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()

    # Step 2 — must parse as JSON
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[VERIFY] JSON parse failed: {e} | First 200 chars: {text[:200]}")
        return [], 1  # whole batch discarded

    # Step 3 — must be a list
    if not isinstance(parsed, list):
        print(f"[VERIFY] Expected list, got {type(parsed).__name__}")
        return [], 1

    expected_label = INTENT_CLASSES[expected_class_id]["label"]
    expected_tier = INTENT_CLASSES[expected_class_id]["tier"]

    for i, item in enumerate(parsed):
        result, reason = _verify_item(item, expected_class_id, expected_label, expected_tier)
        if result:
            valid.append(result)
        else:
            discarded += 1
            print(f"[VERIFY] Item {i} discarded: {reason}")

    return valid, discarded


def _verify_item(item: dict, expected_class_id: int, expected_label: str, expected_tier) -> tuple[dict | None, str]:
    """Verify a single example. Returns (cleaned_item, None) or (None, reason)."""

    if not isinstance(item, dict):
        return None, f"not a dict: {type(item).__name__}"

    # Required fields
    required = ["text", "intent_class_id", "intent_class_label", "language_iso", "resource_class"]
    for field in required:
        if field not in item:
            return None, f"missing field: {field}"

    # text — non-empty string, minimum 3 chars
    text = item.get("text")
    if not isinstance(text, str) or len(text.strip()) < 3:
        return None, f"text too short or not string: {repr(text)}"

    # intent_class_id — must match what was requested exactly
    cid = item.get("intent_class_id")
    if not isinstance(cid, int):
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            return None, f"intent_class_id not int: {repr(cid)}"

    if cid != expected_class_id:
        return None, f"class_id mismatch: got {cid}, expected {expected_class_id}"

    if cid not in VALID_CLASS_IDS:
        return None, f"class_id {cid} not in taxonomy"

    # intent_class_label — must match the canonical label for this ID
    label = item.get("intent_class_label")
    if label != expected_label:
        return None, f"label mismatch: got '{label}', expected '{expected_label}'"

    # language_iso — non-empty string
    lang = item.get("language_iso")
    if not isinstance(lang, str) or len(lang.strip()) == 0:
        return None, f"invalid language_iso: {repr(lang)}"

    # resource_class — must be one of the 5 valid values
    rc = item.get("resource_class")
    if rc not in VALID_RESOURCE_CLASSES:
        return None, f"invalid resource_class: '{rc}' not in {VALID_RESOURCE_CLASSES}"

    # All good — return cleaned record with canonical fields
    return {
        "text": text.strip(),
        "intent_class_id": cid,
        "intent_class_label": expected_label,
        "tier": str(expected_tier),
        "language_iso": lang.strip().lower(),
        "resource_class": rc,
        "generated_by": item.get("generated_by", "unknown"),
        "split": "train",
    }, None


def clean_text_for_jsonl(text: str) -> str:
    """Ensure text is safe for single-line JSONL storage."""
    return text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()

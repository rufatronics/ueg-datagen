"""
Verification — JSON-only gate.
Handles both single-class and mixed-class responses.
Partial recovery — keeps valid items even if some fail.
"""

import json
import re
from taxonomy import INTENT_CLASSES, VALID_CLASS_IDS, ID_TO_LABEL


def verify_batch(raw: str, expected_class_id: int = None) -> tuple[list[dict], int]:
    """
    Verify a raw model response.
    If expected_class_id is set: single-class mode — all items must match.
    If None: mixed-class mode — each item verified against its own class_id.
    Returns (valid_examples, discard_count).
    """
    text = _clean_response(raw)

    # Parse JSON
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        # Try to salvage partial JSON
        parsed = _try_salvage(text)
        if parsed is None:
            print(f"[VERIFY] Parse failed: {e} | snippet: {text[:150]}")
            return [], 1

    if not isinstance(parsed, list):
        print(f"[VERIFY] Expected list, got {type(parsed).__name__}")
        return [], 1

    valid    = []
    discards = 0

    for i, item in enumerate(parsed):
        ok, reason = _verify_item(item, expected_class_id)
        if ok:
            valid.append(_normalize(item))
        else:
            discards += 1
            if discards <= 3:  # avoid log spam
                print(f"[VERIFY] Item {i} discarded: {reason}")

    return valid, discards


def _clean_response(text: str) -> str:
    """Strip markdown fences and whitespace."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If it doesn't start with [ try to find the array
    if not text.startswith("["):
        match = re.search(r"\[", text)
        if match:
            text = text[match.start():]

    # If it doesn't end with ] try to find the last ]
    if not text.endswith("]"):
        match = re.search(r"\](?=[^]]*$)", text)
        if match:
            text = text[:match.end()]

    return text


def _try_salvage(text: str) -> list | None:
    """
    Try to extract as many valid JSON objects as possible from broken output.
    Useful when one bad item in the middle breaks the whole array.
    """
    objects = []
    # Find all {...} blocks and try to parse each
    pattern = re.compile(r'\{[^{}]+\}', re.DOTALL)
    for match in pattern.finditer(text):
        try:
            obj = json.loads(match.group())
            objects.append(obj)
        except Exception:
            continue
    return objects if objects else None


def _verify_item(item: dict, expected_class_id: int = None) -> tuple[bool, str]:
    """Verify a single example dict. Returns (is_valid, reason_if_not)."""

    if not isinstance(item, dict):
        return False, f"not a dict: {type(item).__name__}"

    # Required fields
    for field in ("text", "intent_class_id", "intent_class_label", "language_iso", "resource_class"):
        if field not in item:
            return False, f"missing field: {field}"

    # text — non-empty string, at least 3 chars
    text = item.get("text")
    if not isinstance(text, str) or len(text.strip()) < 3:
        return False, f"text too short or invalid: {repr(text)[:50]}"

    # intent_class_id — must be int and valid
    cid = item.get("intent_class_id")
    if not isinstance(cid, int):
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            return False, f"class_id not int: {repr(cid)}"

    if cid not in VALID_CLASS_IDS:
        return False, f"class_id {cid} not in taxonomy"

    # If single-class mode, enforce expected class
    if expected_class_id is not None and cid != expected_class_id:
        return False, f"class_id mismatch: got {cid}, expected {expected_class_id}"

    # intent_class_label — must match canonical label for this ID
    expected_label = ID_TO_LABEL[cid]
    label = item.get("intent_class_label")
    if label != expected_label:
        return False, f"label mismatch: got '{label}', expected '{expected_label}'"

    # language_iso — non-empty string
    lang = item.get("language_iso")
    if not isinstance(lang, str) or len(lang.strip()) == 0:
        return False, f"invalid language_iso: {repr(lang)}"

    # resource_class — must be one of valid values
    valid_rc = {"hr_global", "mr_regional", "lr_emerging", "mul_mix", "noise_nonlinguistic"}
    rc = item.get("resource_class")
    if rc not in valid_rc:
        return False, f"invalid resource_class: '{rc}'"

    return True, ""


def _normalize(item: dict) -> dict:
    """Return a clean, canonical example dict."""
    cid   = int(item["intent_class_id"])
    label = ID_TO_LABEL[cid]
    tier  = str(INTENT_CLASSES[cid]["tier"])
    return {
        "text":              item["text"].strip(),
        "intent_class_id":   cid,
        "intent_class_label": label,
        "tier":              tier,
        "language_iso":      item["language_iso"].strip().lower(),
        "resource_class":    item["resource_class"],
        "generated_by":      item.get("generated_by", "unknown"),
        "split":             "train",
    }

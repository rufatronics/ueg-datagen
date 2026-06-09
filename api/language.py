"""
Language detector — wraps lingua-language-detector.
Detects the 8 languages from the training distribution.
Fast, accurate on short text, <1ms per call.
"""

import logging
from typing import Optional

logger = logging.getLogger("ueg.lang")

# Map lingua language names to ISO 639-1 codes
LINGUA_TO_ISO = {
    "ENGLISH":    "en",
    "ARABIC":     "ar",
    "HINDI":      "hi",
    "FRENCH":     "fr",
    "SPANISH":    "es",
    "CHINESE":    "zh",
    "SWAHILI":    "sw",
    "PORTUGUESE": "pt",
}

# Resource class mapping by ISO
ISO_TO_RESOURCE = {
    "en": "hr_global",
    "fr": "hr_global",
    "es": "hr_global",
    "zh": "hr_global",
    "pt": "hr_global",
    "ar": "mr_regional",
    "hi": "mr_regional",
    "sw": "lr_emerging",
}

_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
            languages = [
                Language.ENGLISH,   Language.ARABIC,
                Language.HINDI,     Language.FRENCH,
                Language.SPANISH,   Language.CHINESE,
                Language.SWAHILI,   Language.PORTUGUESE,
            ]
            _detector = LanguageDetectorBuilder \
                .from_languages(*languages) \
                .with_minimum_relative_distance(0.1) \
                .build()
            logger.info("Lingua detector initialized")
        except Exception as e:
            logger.warning(f"Lingua not available: {e} — language detection disabled")
            _detector = False
    return _detector if _detector is not False else None


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect language of input text.
    Returns (iso_code, confidence).
    Falls back to 'en' with 0.0 confidence if detection fails.
    """
    if not text or len(text.strip()) < 3:
        return "en", 0.0

    detector = _get_detector()
    if detector is None:
        return "en", 0.0

    try:
        result = detector.detect_language_of(text)
        if result is None:
            return "en", 0.0

        lang_name = result.name  # e.g. "ARABIC"
        iso       = LINGUA_TO_ISO.get(lang_name, "en")

        # Get confidence
        confidence_values = detector.compute_language_confidence_values(text)
        confidence = 0.0
        for cv in confidence_values:
            if cv.language == result:
                confidence = round(cv.value, 4)
                break

        return iso, confidence

    except Exception as e:
        logger.debug(f"Language detection failed: {e}")
        return "en", 0.0


def iso_to_resource_class(iso: str) -> str:
    """Map ISO code to resource class."""
    return ISO_TO_RESOURCE.get(iso, "hr_global")

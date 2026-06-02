"""
UEG Intent Class Taxonomy — single source of truth.
All routing, language distribution, and model config derived from here.
"""

# ---------------------------------------------------------------------------
# Intent classes
# ---------------------------------------------------------------------------

INTENT_CLASSES = {
    1:  {"label": "noise_gibberish",          "tier": 1},
    2:  {"label": "adversarial_probe",         "tier": 1},
    3:  {"label": "greeting_open",             "tier": 2},
    4:  {"label": "phatic_social",             "tier": 2},
    5:  {"label": "closure_gratitude",         "tier": 2},
    6:  {"label": "ui_command",                "tier": 3},
    7:  {"label": "ambient_device_query",      "tier": 3},
    8:  {"label": "navigation_intent",         "tier": 3},
    9:  {"label": "factoid_static",            "tier": 4},
    10: {"label": "factoid_dynamic",           "tier": 4},
    11: {"label": "transactional_status",      "tier": 4},
    12: {"label": "casual_open_chat",          "tier": 4},
    13: {"label": "code_task",                 "tier": "5A"},
    14: {"label": "data_structured",           "tier": "5A"},
    15: {"label": "document_structured",       "tier": "5A"},
    16: {"label": "math_formal",               "tier": "5A"},
    17: {"label": "analysis_reasoning",        "tier": "5B"},
    18: {"label": "long_form_creative",        "tier": "5B"},
    19: {"label": "domain_specialist",         "tier": "5B"},
    20: {"label": "instruction_procedural",    "tier": "5B"},
    21: {"label": "debate_opinion",            "tier": "5B"},
    22: {"label": "multilingual_task",         "tier": "5B"},
}

# ---------------------------------------------------------------------------
# Language distribution
# 70% English, 30% spread across 7 high-signal languages
# Enough for the classifier to learn intent is language-agnostic
# ---------------------------------------------------------------------------

LANGUAGE_POOL = [
    # (iso_code, resource_class, weight)
    ("en",  "hr_global",   70),   # English — dominant
    ("ar",  "mr_regional",  5),   # Arabic
    ("hi",  "mr_regional",  5),   # Hindi
    ("fr",  "hr_global",    5),   # French
    ("es",  "hr_global",    5),   # Spanish
    ("zh",  "hr_global",    4),   # Chinese Simplified
    ("sw",  "lr_emerging",  3),   # Swahili
    ("pt",  "hr_global",    3),   # Portuguese
]

# Pre-build weighted list for fast random.choice()
WEIGHTED_LANGUAGES = []
for iso, rc, weight in LANGUAGE_POOL:
    WEIGHTED_LANGUAGES.extend([(iso, rc)] * weight)

# noise_gibberish and adversarial_probe — English only, language is irrelevant
ENGLISH_ONLY_CLASSES = {1, 2, 13, 14, 15, 16}  # also code/data/math — English makes more sense

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Groq — confirmed current from console.groq.com/docs/rate-limits
GROQ_MODELS = [
    "llama-3.1-8b-instant",                       # 30 RPM, 14.4K RPD — volume horse
    "llama-3.3-70b-versatile",                     # 30 RPM, 1K RPD — quality
    "meta-llama/llama-4-scout-17b-16e-instruct",   # 30 RPM, 1K RPD — balanced
]

# Groq RPD limits per model — used for daily budget tracking
GROQ_RPD = {
    "llama-3.1-8b-instant":                      14400,
    "llama-3.3-70b-versatile":                    1000,
    "meta-llama/llama-4-scout-17b-16e-instruct":  1000,
}

# Gemini — confirmed from AI Studio rate limit dashboard
GEMINI_MODELS = {
    "gemma4_26b":    "gemma-4-26b-a4b-it",    # 15 RPM, 1.5K RPD, UNLIMITED TPM
    "gemma4_31b":    "gemma-4-31b-it",         # 15 RPM, 1.5K RPD, UNLIMITED TPM
    "flash_lite":    "gemini-3.1-flash-lite",  # 15 RPM, 500 RPD
    "flash":         "gemini-3.5-flash",       # 5 RPM,  20 RPD  — use sparingly
}

GEMINI_RPD = {
    "gemma-4-26b-a4b-it":   1500,
    "gemma-4-31b-it":       1500,
    "gemini-3.1-flash-lite": 500,
    "gemini-3.5-flash":       20,
}

GEMINI_RPM = {
    "gemma-4-26b-a4b-it":   15,
    "gemma-4-31b-it":       15,
    "gemini-3.1-flash-lite": 15,
    "gemini-3.5-flash":       5,
}

# Mistral
MISTRAL_MODELS = {
    "large": "mistral-large-latest",
    "small": "mistral-small-latest",
}

# ---------------------------------------------------------------------------
# Batch sizes per provider
# ---------------------------------------------------------------------------

BATCH_SIZES = {
    "groq":          20,   # single class per call
    "gemma4":        80,   # mixed classes per call — unlimited TPM
    "gemini_flash":  20,   # single class per call
    "mistral":       15,   # single class per call
}

# ---------------------------------------------------------------------------
# Target examples per class
# ---------------------------------------------------------------------------

TARGET_PER_CLASS = 8000

VALID_CLASS_IDS   = set(INTENT_CLASSES.keys())
VALID_CLASS_LABELS = {v["label"] for v in INTENT_CLASSES.values()}
ID_TO_LABEL = {k: v["label"] for k, v in INTENT_CLASSES.items()}
LABEL_TO_ID = {v["label"]: k for k, v in INTENT_CLASSES.items()}

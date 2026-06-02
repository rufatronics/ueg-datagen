"""
UEG Intent Class Taxonomy + Language Resource Classes
Single source of truth — all routing logic derives from here.
"""

INTENT_CLASSES = {
    1:  {"label": "noise_gibberish",         "tier": 1,   "description": "Non-semantic character sequences with no detectable linguistic structure."},
    2:  {"label": "adversarial_probe",        "tier": 1,   "description": "Strings exhibiting injection patterns, jailbreak scaffolding, role-override attempts, or known exploit syntax."},
    3:  {"label": "greeting_open",            "tier": 2,   "description": "Simple session-opening salutations."},
    4:  {"label": "phatic_social",            "tier": 2,   "description": "Relational small-talk with no information request."},
    5:  {"label": "closure_gratitude",        "tier": 2,   "description": "Session-closing, acknowledgment, or thanks."},
    6:  {"label": "ui_command",               "tier": 3,   "description": "Direct application interface instructions."},
    7:  {"label": "ambient_device_query",     "tier": 3,   "description": "Environmental or device-state lookups requiring no language model."},
    8:  {"label": "navigation_intent",        "tier": 3,   "description": "User wants to move to a specific location within the application."},
    9:  {"label": "factoid_static",           "tier": 4,   "description": "Single-hop, stable factual retrieval. Answer unlikely to change."},
    10: {"label": "factoid_dynamic",          "tier": 4,   "description": "Single-hop retrieval but answer is time-sensitive. Requires live lookup."},
    11: {"label": "transactional_status",     "tier": 4,   "description": "User querying the state of a specific record or transaction."},
    12: {"label": "casual_open_chat",         "tier": 4,   "description": "Open-ended, unstructured conversational input with no information demand."},
    13: {"label": "code_task",                "tier": "5A", "description": "Any coding input: write, debug, review, refactor, explain code."},
    14: {"label": "data_structured",          "tier": "5A", "description": "SQL queries, data transformation, spreadsheet logic, schema design, regex construction."},
    15: {"label": "document_structured",      "tier": "5A", "description": "Form filling, template generation, structured report writing, contract drafting."},
    16: {"label": "math_formal",              "tier": "5A", "description": "Formal mathematical derivation, proofs, symbolic computation, quantitative problem solving."},
    17: {"label": "analysis_reasoning",       "tier": "5B", "description": "Multi-step comparison, evaluation, causal analysis, strategic thinking."},
    18: {"label": "long_form_creative",       "tier": "5B", "description": "Essays, narratives, scripts, poems, marketing copy, ideation."},
    19: {"label": "domain_specialist",        "tier": "5B", "description": "Medical, legal, financial, scientific, or clinical inputs requiring domain-specific depth."},
    20: {"label": "instruction_procedural",   "tier": "5B", "description": "Step-by-step how-to guides, tutorials, technical walkthroughs."},
    21: {"label": "debate_opinion",           "tier": "5B", "description": "Requests for a position, argument, or perspective."},
    22: {"label": "multilingual_task",        "tier": "5B", "description": "The user's request is itself a language task: translation, grammar correction, cross-lingual summarization."},
}

RESOURCE_CLASSES = {
    "hr_global":        {"langs": ["en", "zh", "es", "fr", "de", "it", "nl", "pl"],               "reliability": "high"},
    "mr_regional":      {"langs": ["ar", "hi", "pt", "ru", "ja", "ko", "tr", "vi", "th", "id"],   "reliability": "medium"},
    "lr_emerging":      {"langs": ["ha", "sw", "yo", "ig", "zu", "am", "so", "rw", "lg", "ff"],   "reliability": "low"},
    "mul_mix":          {"langs": ["pcm", "hinglish", "camfranglais", "spanglish"],                "reliability": "mixed"},
    "noise_nonlinguistic": {"langs": [],                                                            "reliability": "noise"},
}

# Provider routing — who generates what
GROQ_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]          # Tier 1-4, adversarial
MISTRAL_CLASSES = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]       # Tier 5A + 5B English
GEMINI_FLASH_LITE_CLASSES = [17, 18, 19, 20, 21, 22]             # Tier 5B English overflow
GEMINI_FLASH_LANG_CLASSES = list(range(1, 23))                    # All classes — non-English
GEMINI_PRO_CLASSES = [17, 18, 19, 21]                             # lr_emerging + Tier 5B hardest combos

# Language routing
GEMINI_HANDLES_LANGS = ["ha", "sw", "yo", "ig", "zu", "am", "so", "rw", "lg", "ff",
                         "pcm", "hinglish", "camfranglais", "spanglish",
                         "ar", "hi", "zh", "ja", "ko", "tr", "vi", "th", "id",
                         "pt", "ru", "fr", "de", "es", "it", "nl", "pl"]

# Target per class before generation stops for that class
TARGET_PER_CLASS = 8000

# Groq models to rotate across (limits tracked per model independently)
# Current as of June 2026 per https://console.groq.com/docs/deprecations
GROQ_MODELS = [
    "llama-3.3-70b-versatile",      # Best quality 70B
    "llama-3.1-8b-instant",          # Fast 8B workhorse
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Newer Llama 4
]

# OpenRouter model for adversarial
OPENROUTER_ADVERSARIAL_MODEL = "venice-ai/venice-uncensored"

GEMINI_MODELS = {
    "flash_lite": "gemini-2.0-flash-lite",
    "flash":      "gemini-2.5-flash-preview-05-20",
    "pro":        "gemini-2.5-pro-preview-05-06",
}

MISTRAL_MODELS = {
    "large": "mistral-large-latest",
    "small": "mistral-small-latest",
}

VALID_CLASS_IDS = set(INTENT_CLASSES.keys())
VALID_RESOURCE_CLASSES = set(RESOURCE_CLASSES.keys())

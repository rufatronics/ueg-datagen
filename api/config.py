"""
UEG API — Configuration
All constants derived from taxonomy and training config.
No secrets needed — model repo is public.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# HuggingFace — public repo, no token needed
MODEL_REPO = os.getenv("MODEL_REPO", "rufatronics/ueg-classifier")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Model architecture (matches training config exactly)
VOCAB_SIZE   = 32000
MAX_SEQ_LEN  = 128
HIDDEN_DIM   = 512
NUM_LAYERS   = 6
NUM_HEADS    = 8
FFN_DIM      = 2048
DROPOUT      = 0.1
NUM_INTENT   = 22
NUM_RESOURCE = 5

# Intent classes — 0-indexed model output → taxonomy info
INTENT_CLASSES = {
    0:  {"id": 1,  "label": "noise_gibberish",        "tier": "1"},
    1:  {"id": 2,  "label": "adversarial_probe",       "tier": "1"},
    2:  {"id": 3,  "label": "greeting_open",           "tier": "2"},
    3:  {"id": 4,  "label": "phatic_social",           "tier": "2"},
    4:  {"id": 5,  "label": "closure_gratitude",       "tier": "2"},
    5:  {"id": 6,  "label": "ui_command",              "tier": "3"},
    6:  {"id": 7,  "label": "ambient_device_query",    "tier": "3"},
    7:  {"id": 8,  "label": "navigation_intent",       "tier": "3"},
    8:  {"id": 9,  "label": "factoid_static",          "tier": "4"},
    9:  {"id": 10, "label": "factoid_dynamic",         "tier": "4"},
    10: {"id": 11, "label": "transactional_status",    "tier": "4"},
    11: {"id": 12, "label": "casual_open_chat",        "tier": "4"},
    12: {"id": 13, "label": "code_task",               "tier": "5A"},
    13: {"id": 14, "label": "data_structured",         "tier": "5A"},
    14: {"id": 15, "label": "document_structured",     "tier": "5A"},
    15: {"id": 16, "label": "math_formal",             "tier": "5A"},
    16: {"id": 17, "label": "analysis_reasoning",      "tier": "5B"},
    17: {"id": 18, "label": "long_form_creative",      "tier": "5B"},
    18: {"id": 19, "label": "domain_specialist",       "tier": "5B"},
    19: {"id": 20, "label": "instruction_procedural",  "tier": "5B"},
    20: {"id": 21, "label": "debate_opinion",          "tier": "5B"},
    21: {"id": 22, "label": "multilingual_task",       "tier": "5B"},
}

# Resource density classes
RESOURCE_CLASSES = {
    0: "hr_global",
    1: "mr_regional",
    2: "lr_emerging",
    3: "mul_mix",
    4: "noise_nonlinguistic",
}

# Routing action per tier
ROUTING_ACTIONS = {
    "1":  {
        "noise_gibberish":   "drop",
        "adversarial_probe": "block",
    },
    "2":  "static_template",
    "3":  "device_api",
    "4":  "cache_lookup",
    "5A": "route_to_frontier",
    "5B": "route_to_frontier",
}

def get_routing_action(tier: str, label: str) -> str:
    action = ROUTING_ACTIONS.get(tier)
    if isinstance(action, dict):
        return action.get(label, "drop")
    return action or "route_to_frontier"

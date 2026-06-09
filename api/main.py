"""
UEG REST API
POST /classify — classify a single text
POST /classify/batch — classify multiple texts
GET  /health — health check
GET  /info — model info
"""

import time
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import (
    MODEL_REPO,
    INTENT_CLASSES, RESOURCE_CLASSES,
    get_routing_action, HOST, PORT,
)
from model import engine
from language import detect_language, iso_to_resource_class

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ueg.api")


# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("UEG API starting — loading model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.load)
    logger.info("Model loaded — API ready")
    yield
    logger.info("UEG API shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="UEG — Universal Edge Gateway",
    description=(
        "Intent classifier for AI routing. Classifies incoming text into "
        "22 intent classes and 5 language resource density classes in <5ms."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Raw user input to classify", min_length=0)
    include_probabilities: bool = Field(
        False, description="Include full probability distribution in response"
    )


class ClassifyResponse(BaseModel):
    # Intent
    intent_class_id:    int
    intent_class_label: str
    tier:               str
    routing_action:     str
    confidence_intent:  float

    # Resource density
    resource_class:          str
    confidence_resource:     float

    # Language
    language_iso:            str
    language_confidence:     float

    # Meta
    latency_ms:              float
    model:                   str = "ueg-classifier-v1"

    # Optional
    probabilities_intent:    Optional[list[float]] = None
    probabilities_resource:  Optional[list[float]] = None


class BatchClassifyRequest(BaseModel):
    texts:                 list[str] = Field(..., max_length=64)
    include_probabilities: bool = False


class HealthResponse(BaseModel):
    status:  str
    model:   str
    ready:   bool


# ---------------------------------------------------------------------------
# Core classify function
# ---------------------------------------------------------------------------
def _classify_text(text: str, include_probs: bool = False) -> dict:
    t_start = time.perf_counter()

    # Empty input gate
    if not text or not text.strip():
        return {
            "intent_class_id":    1,
            "intent_class_label": "noise_gibberish",
            "tier":               "1",
            "routing_action":     "drop",
            "confidence_intent":  1.0,
            "resource_class":     "noise_nonlinguistic",
            "confidence_resource": 1.0,
            "language_iso":       "en",
            "language_confidence": 0.0,
            "latency_ms":         round((time.perf_counter() - t_start) * 1000, 3),
            "model":              "ueg-classifier-v1",
        }

    # Language detection
    lang_iso, lang_conf = detect_language(text.strip())

    # Model inference
    result = engine.infer(text.strip())

    # Decode intent
    intent_idx  = result["intent_idx"]
    intent_info = INTENT_CLASSES[intent_idx]

    # Decode resource
    resource_idx   = result["resource_idx"]
    resource_label = RESOURCE_CLASSES[resource_idx]

    # Routing action
    routing = get_routing_action(intent_info["tier"], intent_info["label"])

    latency = round((time.perf_counter() - t_start) * 1000, 3)

    response = {
        "intent_class_id":    intent_info["id"],
        "intent_class_label": intent_info["label"],
        "tier":               intent_info["tier"],
        "routing_action":     routing,
        "confidence_intent":  round(result["confidence_intent"], 4),
        "resource_class":     resource_label,
        "confidence_resource": round(result["confidence_resource"], 4),
        "language_iso":       lang_iso,
        "language_confidence": lang_conf,
        "latency_ms":         latency,
        "model":              "ueg-classifier-v1",
    }

    if include_probs:
        response["probabilities_intent"]   = [round(p, 4) for p in result["probs_intent"]]
        response["probabilities_resource"] = [round(p, 4) for p in result["probs_resource"]]

    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return {
        "status": "ok" if engine.ready else "loading",
        "model":  MODEL_REPO,
        "ready":  engine.ready,
    }


@app.get("/info", tags=["System"])
async def info():
    return {
        "model":        "ueg-classifier-v1",
        "repo":         MODEL_REPO,
        "parameters":   "35M",
        "intent_classes": {
            v["id"]: {"label": v["label"], "tier": v["tier"]}
            for v in INTENT_CLASSES.values()
        },
        "resource_classes": RESOURCE_CLASSES,
        "routing_actions": {
            "drop":              "Tier 1 noise — discard silently",
            "block":             "Tier 1 adversarial — reject with warning",
            "static_template":   "Tier 2 — respond with pre-written template",
            "device_api":        "Tier 3 — call local device/environment API",
            "cache_lookup":      "Tier 4 — check cache or micro-LLM",
            "route_to_frontier": "Tier 5 — send to full frontier model",
        },
        "max_seq_len": 128,
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
async def classify(req: ClassifyRequest):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _classify_text, req.text, req.include_probabilities
        )
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", tags=["Inference"])
async def classify_batch(req: BatchClassifyRequest):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    if not req.texts:
        raise HTTPException(status_code=400, detail="texts list is empty")

    try:
        loop    = asyncio.get_event_loop()
        results = []
        t_batch = time.perf_counter()

        for text in req.texts:
            result = await loop.run_in_executor(
                None, _classify_text, text, req.include_probabilities
            )
            results.append(result)

        total_ms = round((time.perf_counter() - t_batch) * 1000, 3)

        return {
            "results":      results,
            "count":        len(results),
            "total_ms":     total_ms,
            "avg_ms":       round(total_ms / len(results), 3),
        }
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)

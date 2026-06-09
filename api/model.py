"""
UEG Model Loader
Downloads ONNX model + tokenizer from HuggingFace on first startup.
Public repo — no token needed.
Caches locally so subsequent restarts are instant.
"""

import os
import json
import shutil
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from config import MODEL_REPO, MAX_SEQ_LEN

logger    = logging.getLogger("ueg.loader")
CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/ueg-cache"))


class UEGInferenceEngine:
    """
    Wraps ONNX model + tokenizer.
    Thread-safe — onnxruntime sessions are stateless per inference call.
    """

    def __init__(self):
        self.session   = None
        self.tokenizer = None
        self.pad_id    = 0
        self._ready    = False

    def load(self):
        """Download and initialize everything. Called once at startup."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading UEG from {MODEL_REPO}...")

        # Download files from public HF repo
        onnx_path = self._download("export/ueg_model.onnx")
        tok_path  = self._download("tokenizer/tokenizer.json")
        cfg_path  = self._download("tokenizer/tokenizer_config.json")

        # Get pad_id from tokenizer config
        with open(cfg_path) as f:
            tok_cfg = json.load(f)
        self.pad_id = tok_cfg["pad_id"]

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_padding(
            pad_id=self.pad_id,
            pad_token="[PAD]",
            length=MAX_SEQ_LEN,
        )
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        logger.info(f"Tokenizer ready — vocab: {self.tokenizer.get_vocab_size():,}")

        # Load ONNX session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads      = int(os.getenv("ORT_THREADS", "2"))

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info(f"ONNX session ready")
        self._ready = True
        logger.info("UEG engine ready ✓")

    def infer(self, text: str) -> dict:
        """Run inference. Returns probs and indices for both heads."""
        if not self._ready:
            raise RuntimeError("Engine not loaded")

        enc  = self.tokenizer.encode(text)
        ids  = enc.ids[:MAX_SEQ_LEN]
        mask = enc.attention_mask[:MAX_SEQ_LEN]
        pad  = MAX_SEQ_LEN - len(ids)
        ids  += [self.pad_id] * pad
        mask += [0] * pad

        outputs = self.session.run(
            ["logits_intent", "logits_resource"],
            {
                "input_ids":      np.array([ids],  dtype=np.int64),
                "attention_mask": np.array([mask], dtype=np.int64),
            },
        )

        def softmax(x):
            e = np.exp(x - np.max(x))
            return e / e.sum()

        pi = softmax(outputs[0][0])
        pr = softmax(outputs[1][0])

        return {
            "intent_idx":          int(np.argmax(pi)),
            "resource_idx":        int(np.argmax(pr)),
            "probs_intent":        pi.tolist(),
            "probs_resource":      pr.tolist(),
            "confidence_intent":   float(pi.max()),
            "confidence_resource": float(pr.max()),
        }

    def _download(self, filename: str) -> Path:
        """Download from public HF repo, cache locally."""
        local = CACHE_DIR / filename.replace("/", "_")
        if local.exists():
            logger.debug(f"Cache hit: {filename}")
            return local

        logger.info(f"Downloading {filename}...")
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            repo_type="model",
            cache_dir=str(CACHE_DIR / "hf_cache"),
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, local)
        return local

    @property
    def ready(self) -> bool:
        return self._ready


# Singleton
engine = UEGInferenceEngine()

# app_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
import logging
import typing
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
MODEL_PATH = "./roberta"             # folder with your model files
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_WORDS = 8
# ----------------------------

app = FastAPI(title="Sentiment RoBERTa API")

# CORS (development). In production, restrict origins to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files under /static (avoid shadowing API routes)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------- load model & tokenizer ----------------
logger.info("Loading tokenizer and model from: %s", MODEL_PATH)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.to(DEVICE)
    model.eval()
    logger.info("Model loaded. Device: %s", DEVICE)

    # Build an explicit index->label mapping from model config if available
    cfg_map = getattr(model.config, "id2label", None)
    if cfg_map and isinstance(cfg_map, dict) and len(cfg_map) > 0:
        IDX2LABEL = {int(k): str(v) for k, v in cfg_map.items()}
    else:
        # safe default mapping (common convention)
        IDX2LABEL = {0: "Negative", 1: "Positive"}

    # Also derive some meta info
    NUM_LABELS = getattr(model.config, "num_labels", None)
    if NUM_LABELS is None:
        # fallback guess from model outputs (rare)
        NUM_LABELS = 2

    IS_SINGLE_LABEL = int(NUM_LABELS) == 1  # regression / sigmoid style model
    logger.info("IDX2LABEL=%s, num_labels=%s, single_label=%s", IDX2LABEL, NUM_LABELS, IS_SINGLE_LABEL)

except Exception as e:
    logger.exception("Failed to load model/tokenizer from %s: %s", MODEL_PATH, e)
    raise


# ---------------- helpers & schemas ----------------
def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.strip().split())


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float
    raw: typing.Optional[dict] = None


# ---------------- routes ----------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if count_words(text) < MIN_WORDS:
        raise HTTPException(status_code=400, detail=f"Input must contain at least {MIN_WORDS} words.")

    # tokenize
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=getattr(tokenizer, "model_max_length", 512),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tokenization error: {e}")

    # move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape (1, num_labels) or (1,1)

    # handle single-label (sigmoid/regression) vs multi-class (softmax)
    try:
        if IS_SINGLE_LABEL:
            # logits is a scalar or shape (1,1)
            logit = logits.squeeze().cpu().item()
            score = float(torch.sigmoid(torch.tensor(logit)).item())
            # decide label using score threshold 0.5 and mapping (IDX2LABEL expects 0/1)
            label = IDX2LABEL.get(1 if score > 0.5 else 0, ("Positive" if score > 0.5 else "Negative"))
            probs = [1.0 - score, score]  # fake two-entry probs for compatibility
            raw_logits = [float(logit)]
        else:
            arr = F.softmax(logits, dim=-1).cpu().squeeze().numpy()
            # normalize shape: ensure 1D array
            arr = np.array(arr).ravel()
            max_idx = int(arr.argmax())
            score = float(arr[max_idx])
            label = IDX2LABEL.get(max_idx, str(max_idx))
            probs = arr.tolist()
            raw_logits = logits.cpu().squeeze().numpy().tolist()
    except Exception as e:
        logger.exception("Error processing model outputs: %s", e)
        raise HTTPException(status_code=500, detail="Error processing model outputs")

    # include idx2label mapping as strings in raw for front-end debugging/robustness
    idx2label_str = {str(k): v for k, v in IDX2LABEL.items()}

    return PredictResponse(
        label=label,
        score=round(score, 6),
        raw={
            "probs": probs,
            "logits": raw_logits,
            "idx2label": idx2label_str,
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


# serve index explicitly at root (do not include in OpenAPI docs)
@app.get("/", include_in_schema=False)
def index():
    # ensure file exists
    index_path = os.path.join("static", "index.html")
    if not os.path.isfile(index_path):
        return {"error": "index.html not found in static/ (expected at static/index.html)"}
    return FileResponse(index_path)


# ---------------- run (development) ----------------
if __name__ == "__main__":
    # development: use reload for convenience. Remove reload in production.
    import uvicorn

    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000, reload=True)

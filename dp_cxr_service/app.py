"""
FastAPI service for the exported DP-CXR model.

Run:
    uvicorn dp_cxr_service.app:app --reload --port 8000

Interactive docs at http://localhost:8000/docs -- upload a radiograph and paste a
report straight from the browser, which makes for a clean viva demonstration
without writing a client.

Endpoints:
    GET  /health        liveness plus which bundle is loaded
    GET  /model-card    full provenance: privacy, performance, limitations
    POST /predict       multipart -- image file and/or report text
    POST /predict-json  JSON -- base64 image and/or report text
"""
from __future__ import annotations

import base64
import io
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel, Field

from .predictor import DPCXRPredictor

BUNDLE_DIR = os.environ.get("DP_CXR_BUNDLE", str(Path(__file__).parent / "bundle"))

app = FastAPI(
    title="DP-CXR — Differentially Private Multimodal Chest X-ray Prediction",
    description=(
        "Multilabel chest-pathology prediction from a frontal radiograph and/or "
        "pre-diagnostic report text, served from a DP-SGD trained model with "
        "Grad-CAM and token-attribution explanations.\n\n"
        "**Not a medical device.** Research demonstration for an MSc dissertation."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Loaded lazily so the process starts even if the bundle is missing -- /health
# then reports the problem instead of the container crash-looping.
_predictor: Optional[DPCXRPredictor] = None
_load_error: Optional[str] = None


def get_predictor() -> DPCXRPredictor:
    global _predictor, _load_error
    if _predictor is None:
        try:
            _predictor = DPCXRPredictor(BUNDLE_DIR)
            _load_error = None
        except Exception as e:
            _load_error = f"{type(e).__name__}: {e}"
            raise HTTPException(
                status_code=503,
                detail=f"Model bundle not loaded ({_load_error}). Unpack "
                       f"dp_cxr_deployment_bundle.zip into {BUNDLE_DIR}.",
            )
    return _predictor


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        get_predictor()                      # warm the weights before first request
    except Exception as e:
        print(f"[startup] bundle not loaded yet: {e}")
    yield


app.router.lifespan_context = lifespan


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────
class PredictJSONRequest(BaseModel):
    image_base64: Optional[str] = Field(
        None, description="Base64 PNG/JPEG. Accepts a raw string or a data URI."
    )
    report_text: Optional[str] = Field(
        None,
        description="Radiology report. Findings and Impression are stripped "
                    "server-side to match training.",
    )
    include_gradcam: bool = True
    include_token_attribution: bool = True
    include_metadata: bool = True
    top_k_tokens: int = 12


def _decode_b64_image(s: str) -> Image.Image:
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(s)))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image_base64: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        p = get_predictor()
    except HTTPException:
        return {"status": "degraded", "bundle_loaded": False, "error": _load_error,
                "bundle_dir": BUNDLE_DIR}
    priv = p.card.get("privacy", {})
    return {
        "status": "ok",
        "bundle_loaded": True,
        "model": p.cfg["run_name"],
        "fusion_type": p.cfg["fusion_type"],
        "labels": p.labels,
        "device": p.device,
        "differentially_private": priv.get("mechanism", "").startswith("DP-SGD"),
        "epsilon": priv.get("achieved_epsilon"),
    }


@app.get("/model-card")
def model_card():
    return get_predictor().card


@app.post("/predict")
async def predict(
    image: Optional[UploadFile] = File(None, description="Frontal chest radiograph"),
    report_text: Optional[str] = Form(None, description="Radiology report text"),
    include_gradcam: bool = Form(True),
    include_token_attribution: bool = Form(True),
    include_metadata: bool = Form(True),
):
    """Predict from an uploaded radiograph, report text, or both."""
    p = get_predictor()
    img = None
    if image is not None and image.filename:
        raw = await image.read()
        if not raw:
            raise HTTPException(400, "Uploaded image file is empty.")
        try:
            img = Image.open(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(400, f"Unreadable image: {e}")

    if img is None and not (report_text or "").strip():
        raise HTTPException(400, "Supply at least one of: image, report_text.")

    try:
        return p.predict(
            image=img,
            report_text=report_text,
            include_gradcam=include_gradcam,
            include_token_attribution=include_token_attribution,
            include_metadata=include_metadata,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        # print the whole stack to the terminal running uvicorn; a one-line
        # message in the browser is rarely enough to locate the real cause
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/predict-json")
def predict_json(req: PredictJSONRequest):
    """Same as /predict but JSON in, JSON out."""
    p = get_predictor()
    img = _decode_b64_image(req.image_base64) if req.image_base64 else None
    if img is None and not (req.report_text or "").strip():
        raise HTTPException(400, "Supply at least one of: image_base64, report_text.")
    try:
        return p.predict(
            image=img,
            report_text=req.report_text,
            include_gradcam=req.include_gradcam,
            include_token_attribution=req.include_token_attribution,
            include_metadata=req.include_metadata,
            top_k_tokens=req.top_k_tokens,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the single-page frontend.

    Read from disk on each request rather than cached at import, so editing the
    HTML only needs a browser refresh, not a server restart.
    """
    page = STATIC / "index.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>DP-CXR</h1><p>The frontend file is missing. Expected it at "
        f"<code>{page}</code>.</p><p><a href='/docs'>Use the API docs instead</a>.</p>",
        status_code=200,
    )

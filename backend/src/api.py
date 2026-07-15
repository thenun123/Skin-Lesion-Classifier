"""
FastAPI application entrypoint.

Run locally:
    uvicorn api:app --reload --port 8000

Run via Docker:
    docker compose up --build

Endpoints:
    GET  /health   -> model load status — poll this before trusting /predict
    GET  /metrics  -> rolling request / latency / class-distribution summary
    POST /predict  -> multipart image upload -> classification result (see schemas.py)

API docs (auto-generated from Pydantic schemas):
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import config
from inference import predict
from model import ModelNotLoadedError, model_wrapper
from monitoring import logger, monitor
from schemas import HealthResponse, MetricsResponse, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Load the model at startup. Soft-fail so /health can report the problem clearly."""
    try:
        model_wrapper.load()
    except ModelNotLoadedError as e:
        # Don't crash the process — let the API come up so /health can surface the
        # problem instead of the container dying silently.
        logger.error("Model failed to load at startup: %s", e)
    yield


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next: Any) -> Any:
    """Attach a unique X-Request-ID header to every response for traceability."""
    request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model health check",
    tags=["Ops"],
)
def health() -> HealthResponse:
    """Return the model load status and path. Poll this before trusting /predict."""
    return HealthResponse(
        status="ok" if model_wrapper.is_ready() else "model_not_loaded",
        model_path=config.MODEL_PATH,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Rolling prediction metrics",
    tags=["Ops"],
)
def metrics() -> MetricsResponse:
    """Return rolling request count, average latency, and recent malignant rate."""
    return MetricsResponse(**monitor.summary())


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a skin lesion image",
    tags=["Inference"],
)
async def predict_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a JPEG or PNG dermoscopic image and receive a benign/malignant classification.

    Returns the predicted label, raw malignant probability, confidence score,
    decision threshold, and inference latency.
    """
    if not model_wrapper.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Check GET /health and confirm the model "
                "file exists in the models/ directory."
            ),
        )

    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Allowed: {sorted(config.ALLOWED_CONTENT_TYPES)}"
            ),
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max is {config.MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        result = predict(contents)
    except Exception as e:
        monitor.record_error(str(e), context="predict_endpoint")
        logging.getLogger("skin_cancer_api").exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    monitor.record_prediction(result, filename=file.filename)
    return JSONResponse(content=result)

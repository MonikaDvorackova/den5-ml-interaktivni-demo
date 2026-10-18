"""FastAPI server pro inference natrénované Iris pipeline.

Lokální spuštění (z kořene repozitáře, po `python -m mlops.train`):

    uvicorn mlops.api.main:app --reload --port 8000

Pak otevři automaticky generovanou dokumentaci na http://127.0.0.1:8000/docs
a vyzkoušej POST /predict přímo z prohlížeče ("Try it out").

Endpointy:
  GET  /health       — je server živý a má nahraný model?
  GET  /model-info    — metadata a metriky aktivní verze modelu
  POST /predict       — predikce pro jeden květ Iris

Model se načítá JEDNOU při startu serveru (`lifespan`/startup) — žádný
request neprovádí trénink ani přetrénování, pouze `predict()`/`predict_proba()`
na už naučené pipeline. To je klíčový rozdíl mezi trénovací a inferenční
pipeline: trénink běží offline (viz `mlops/train.py`), server za běhu jen
obsluhuje predikce.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from mlops.model import load_pipeline, predict_one
from mlops.schemas import (
    HealthResponse,
    IrisFeatures,
    ModelInfoResponse,
    PredictionResponse,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Kterou verzi modelu server nasadí, lze přepnout proměnnou prostředí
# MODEL_VERSION (např. při rolloutu v2 nebo rollbacku zpět na v1) —
# bez nutnosti měnit kód.
ACTIVE_VERSION = os.environ.get("MODEL_VERSION", "v1")

_state: dict[str, Any] = {"pipeline": None, "metadata": None}


def _artifact_path(version: str) -> Path:
    return ARTIFACTS_DIR / f"model_{version}.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = _artifact_path(ACTIVE_VERSION)
    if not path.exists():
        raise RuntimeError(
            f"Artefakt modelu '{path}' neexistuje. Spusť nejprve `python -m mlops.train`."
        )
    pipeline, metadata = load_pipeline(str(path))
    _state["pipeline"] = pipeline
    _state["metadata"] = metadata
    yield
    _state["pipeline"] = None
    _state["metadata"] = None


app = FastAPI(
    title="Iris Classifier API",
    description="Výukové FastAPI pro predikci druhu Iris — serving natrénované scikit-learn pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state["pipeline"] is not None else "model_not_loaded",
        model_loaded=_state["pipeline"] is not None,
        model_version=ACTIVE_VERSION,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["monitoring"])
def model_info() -> ModelInfoResponse:
    if _state["metadata"] is None:
        raise HTTPException(status_code=503, detail="Model ještě není nahraný.")
    md = _state["metadata"]
    return ModelInfoResponse(
        model_version=md["version"],
        description=md["description"],
        dataset_version=md["dataset_version"],
        hyperparameters=md["hyperparameters"],
        metrics=md.get("metrics", {}),
        feature_names=md["feature_names"],
        target_names=md["target_names"],
        trained_at_utc=md["trained_at_utc"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: IrisFeatures) -> PredictionResponse:
    if _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model ještě není nahraný.")
    result = predict_one(_state["pipeline"], features.to_list())
    return PredictionResponse(
        prediction=result["prediction"],
        class_name=result["class_name"],
        probabilities=result["probabilities"],
        model_version=ACTIVE_VERSION,
    )

"""Sdílená logika pro trénink, evaluaci a serializaci Iris pipeline.

Používá se jak ze Streamlit stránek (modul "ML Deployment & MLOps"), tak
z FastAPI serveru (`mlops/api/main.py`) a z trénovacího skriptu
(`mlops/train.py`) — aby model v demo API a v `train.py` vznikal přesně
stejným způsobem jako ten, co si student "osahá" ve Streamlitu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

# Konfigurace jednotlivých "vydaných" verzí modelu — používá se v demu
# verzování (v1/v2) i jako výchozí konfigurace v train.py.
MODEL_VERSIONS: dict[str, dict[str, Any]] = {
    "v1": {
        "description": "Počáteční verze — výchozí regularizace.",
        "dataset_version": "iris-sklearn-1.9 (150 vzorků, bez úprav)",
        "test_size": 0.2,
        "random_state": 42,
        "C": 1.0,
        "max_iter": 200,
    },
    "v2": {
        "description": "Silnější regularizace (nižší C) po analýze přeučení na train datech.",
        "dataset_version": "iris-sklearn-1.9 (150 vzorků, bez úprav)",
        "test_size": 0.2,
        "random_state": 42,
        "C": 0.2,
        "max_iter": 200,
    },
}


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict[str, Any]
    X_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def load_iris_split(test_size: float = 0.2, random_state: int = 42):
    """Načte Iris dataset a vrátí stratifikovaný train/test split."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target,
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(C: float = 1.0, max_iter: int = 200, random_state: int = 42) -> Pipeline:
    """Sestaví scikit-learn Pipeline: StandardScaler -> LogisticRegression.

    Preprocessing (StandardScaler) je součástí pipeline, takže se vždy
    nafitne JEN na trénovacích datech a na testovacích/nových datech se jen
    použije (transform) — žádný leak informací z testu/produkce do fitu.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_evaluate(
    C: float = 1.0,
    max_iter: int = 200,
    random_state: int = 42,
    test_size: float = 0.2,
) -> TrainResult:
    """Natrénuje pipeline a vrátí ji spolu s metrikami na testovací sadě.

    `fit()` se volá jen tady (trénink). Cokoliv dál v inference/API kódu
    smí už jen `predict()` / `predict_proba()` — bez přetrénování.
    """
    X_train, X_test, y_train, y_test = load_iris_split(test_size=test_size, random_state=random_state)

    pipeline = build_pipeline(C=C, max_iter=max_iter, random_state=random_state)
    pipeline.fit(X_train, y_train)  # <-- fit(): pipeline se "učí" na trénovacích datech

    y_pred = pipeline.predict(X_test)  # <-- predict(): jen použití už naučeného modelu
    y_proba = pipeline.predict_proba(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    metadata = {
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "hyperparameters": {"C": C, "max_iter": max_iter},
        "random_state": random_state,
        "test_size": test_size,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_pipeline_steps": [name for name, _ in pipeline.steps],
    }

    return TrainResult(
        pipeline=pipeline,
        metrics=metrics,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        metadata=metadata,
    )


def save_pipeline(pipeline: Pipeline, metadata: dict[str, Any], path: str) -> None:
    """Uloží pipeline + metadata do jednoho souboru pomocí joblib.

    Ukládáme VLASTNÍ, čerstvě natrénovaný artefakt — joblib.load() dál
    v kódu proto nikdy nenačítá nic, co by pocházelo z nedůvěryhodného
    zdroje (viz `load_pipeline`).
    """
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, path)


def load_pipeline(path: str) -> tuple[Pipeline, dict[str, Any]]:
    """Načte pipeline + metadata uložené funkcí `save_pipeline`.

    Bezpečnostní pozn.: joblib.load používá pickle pod kapotou, a pickle
    umí při deserializaci spustit libovolný kód. Proto tuto funkci volej
    POUZE na soubory, které vznikly v tomto repozitáři (`train.py` /
    `save_pipeline`) — nikdy na artefakt stažený z neznámého zdroje.
    """
    payload = joblib.load(path)
    return payload["pipeline"], payload["metadata"]


def predict_one(pipeline: Pipeline, features: list[float]) -> dict[str, Any]:
    """Jedna predikce pro 4 vstupní rysy Iris -> label, název třídy, pravděpodobnosti."""
    X = np.array([features], dtype=float)
    pred = int(pipeline.predict(X)[0])
    proba = pipeline.predict_proba(X)[0].tolist()
    return {
        "prediction": pred,
        "class_name": TARGET_NAMES[pred],
        "probabilities": {name: float(p) for name, p in zip(TARGET_NAMES, proba)},
    }

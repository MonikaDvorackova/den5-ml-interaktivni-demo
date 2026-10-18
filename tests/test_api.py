"""Automatizované testy FastAPI endpointů (mlops/api/main.py).

Vyžaduje vygenerované artefakty — spusť nejprve `python -m mlops.train`.
"""
import pytest
from fastapi.testclient import TestClient

from mlops.api.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "v1"


def test_model_info_contains_metrics_and_features(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v1"
    assert set(body["feature_names"]) == {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    }
    assert "accuracy" in body["metrics"]


def test_predict_valid_payload_returns_prediction(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["class_name"] in {"setosa", "versicolor", "virginica"}
    assert body["model_version"] == "v1"
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_predict_rejects_negative_feature(client):
    payload = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_feature(client):
    payload = {
        "sepal_length": 500.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

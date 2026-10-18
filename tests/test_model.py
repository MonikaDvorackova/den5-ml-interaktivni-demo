"""Testy tréninku a serializace Iris pipeline (mlops/model.py)."""
import os
import tempfile

from mlops.model import load_pipeline, predict_one, save_pipeline, train_and_evaluate


def test_training_is_reproducible_with_fixed_seed():
    result_a = train_and_evaluate(C=1.0, random_state=42)
    result_b = train_and_evaluate(C=1.0, random_state=42)
    assert result_a.metrics["accuracy"] == result_b.metrics["accuracy"]
    assert (result_a.y_pred == result_b.y_pred).all()


def test_scaler_is_fit_only_on_training_data():
    result = train_and_evaluate(C=1.0, random_state=42)
    scaler = result.pipeline.named_steps["scaler"]
    # StandardScaler fitnutý na 120 trénovacích vzorcích (150 - 20% test split).
    assert scaler.n_samples_seen_ == 120


def test_metrics_are_within_valid_ranges():
    result = train_and_evaluate(C=1.0, random_state=42)
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert 0.0 <= result.metrics[key] <= 1.0
    assert result.metrics["n_train"] + result.metrics["n_test"] == 150


def test_save_and_reload_pipeline_predicts_identically():
    result = train_and_evaluate(C=1.0, random_state=42)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "model.joblib")
        save_pipeline(result.pipeline, result.metadata, path)
        reloaded_pipeline, reloaded_metadata = load_pipeline(path)

        sample = result.X_test[0].tolist()
        original = predict_one(result.pipeline, sample)
        reloaded = predict_one(reloaded_pipeline, sample)

        assert original == reloaded
        assert reloaded_metadata["hyperparameters"]["C"] == 1.0


def test_predict_one_returns_valid_class_name():
    result = train_and_evaluate(C=1.0, random_state=42)
    prediction = predict_one(result.pipeline, [5.1, 3.5, 1.4, 0.2])
    assert prediction["class_name"] in {"setosa", "versicolor", "virginica"}
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-6

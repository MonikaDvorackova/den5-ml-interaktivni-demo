"""Trénovací skript — vygeneruje artefakty modelu použité FastAPI serverem
a demem verzování ve Streamlitu.

Spuštění (z kořene repozitáře `den5_interaktivni_app`):

    python -m mlops.train

Vytvoří `mlops/artifacts/model_v1.joblib` a `model_v2.joblib` (pipeline +
metadata) a `mlops/artifacts/registry.json` (souhrn verzí a metrik, čte ho
Streamlit stránka s verzováním beze nutnosti importovat scikit-learn).
"""
from __future__ import annotations

import json
from pathlib import Path

from mlops.model import MODEL_VERSIONS, save_pipeline, train_and_evaluate

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    registry: dict[str, dict] = {}

    for version, cfg in MODEL_VERSIONS.items():
        result = train_and_evaluate(
            C=cfg["C"],
            max_iter=cfg["max_iter"],
            random_state=cfg["random_state"],
            test_size=cfg["test_size"],
        )
        metadata = {
            **result.metadata,
            "version": version,
            "description": cfg["description"],
            "dataset_version": cfg["dataset_version"],
            "metrics": result.metrics,
        }
        artifact_path = ARTIFACTS_DIR / f"model_{version}.joblib"
        save_pipeline(result.pipeline, metadata, str(artifact_path))

        registry[version] = {
            "description": cfg["description"],
            "dataset_version": cfg["dataset_version"],
            "hyperparameters": metadata["hyperparameters"],
            "metrics": result.metrics,
            "trained_at_utc": metadata["trained_at_utc"],
            "artifact": artifact_path.name,
        }
        print(f"[train] uložen {artifact_path} (accuracy={result.metrics['accuracy']:.3f})")

    registry_path = ARTIFACTS_DIR / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[train] registry uložen do {registry_path}")


if __name__ == "__main__":
    main()

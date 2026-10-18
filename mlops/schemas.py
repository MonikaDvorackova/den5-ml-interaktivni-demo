"""Pydantic modely pro validaci vstupu/výstupu FastAPI serveru.

Sdílené mezi `mlops/api/main.py` a Streamlit demo stránkami, aby JSON
ukázaný studentům ve Streamlitu přesně odpovídal tomu, co server opravdu
validuje.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    """Vstup pro POST /predict — čtyři rysy květu Iris v centimetrech."""

    sepal_length: float = Field(..., gt=0, le=15, description="Délka kalicha (cm)")
    sepal_width: float = Field(..., gt=0, le=15, description="Šířka kalicha (cm)")
    petal_length: float = Field(..., gt=0, le=15, description="Délka okvětního lístku (cm)")
    petal_width: float = Field(..., gt=0, le=15, description="Šířka okvětního lístku (cm)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    }

    def to_list(self) -> list[float]:
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Číselný label predikované třídy (0, 1, 2)")
    class_name: str = Field(..., description="Název predikované třídy (setosa/versicolor/virginica)")
    probabilities: dict[str, float] = Field(..., description="Pravděpodobnost pro každou třídu")
    model_version: str = Field(..., description="Verze modelu, která predikci vytvořila")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    description: str
    dataset_version: str
    hyperparameters: dict
    metrics: dict
    feature_names: list[str]
    target_names: list[str]
    trained_at_utc: str

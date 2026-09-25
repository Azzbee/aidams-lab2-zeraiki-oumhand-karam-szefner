"""Row-wise features and validated batch inference for the Lab 2 pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
from numpy.typing import NDArray

BASE_NUMERIC = [
    "steel_capacity_ttpa",
    "iron_capacity_ttpa",
    "eaf_capacity_ttpa",
    "workforce",
    "plant_age",
]
CATEGORICAL = ["country", "equipment", "active_status"]
ENGINEERED = ["log_steel_capacity", "capacity_per_worker", "eaf_share"]
FEATURES = BASE_NUMERIC + CATEGORICAL


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute features per row without learning from training or test samples."""
    result = frame.copy()
    result["log_steel_capacity"] = np.log1p(result["steel_capacity_ttpa"])
    result["capacity_per_worker"] = result["steel_capacity_ttpa"].div(
        result["workforce"].where(result["workforce"] > 0)
    )
    result["eaf_share"] = result["eaf_capacity_ttpa"].div(
        result["steel_capacity_ttpa"].where(result["steel_capacity_ttpa"] > 0)
    )
    return result.replace([np.inf, -np.inf], np.nan)


def input_schema() -> pa.DataFrameSchema:
    """Validate normalized snapshot inputs; unknown numerics remain missing."""
    columns = {
        name: pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True)
        for name in BASE_NUMERIC
    }
    columns.update({name: pa.Column(str, nullable=False) for name in CATEGORICAL})
    columns["active_status"] = pa.Column(
        str,
        pa.Check.isin(
            ["operating", "operating pre-retirement", "both", "no active capacity"]
        ),
    )
    return pa.DataFrameSchema(columns, strict=True)


def predict_batch(frame: pd.DataFrame, artifact_dir: Path) -> NDArray[np.float64]:
    """Validate normalized inputs and run the complete, trusted local pipeline."""
    validated = input_schema().validate(frame, lazy=True)
    pipeline = joblib.load(artifact_dir / "best_pipeline.joblib")
    return np.asarray(pipeline.predict(validated), dtype=float)

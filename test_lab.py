"""Check feature edge cases, serving validation and saved-model parity."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from lab_utils import FEATURES, engineer_features, input_schema, predict_batch

BASE = Path(__file__).resolve().parent


def example() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "steel_capacity_ttpa": [1000.0, 0.0, np.nan],
            "iron_capacity_ttpa": [800.0, 0.0, np.nan],
            "eaf_capacity_ttpa": [400.0, 0.0, np.nan],
            "workforce": [500.0, 0.0, np.nan],
            "plant_age": [20.0, 5.0, np.nan],
            "country": ["Example"] * 3,
            "equipment": ["EAF"] * 3,
            "active_status": ["operating", "no active capacity", "both"],
        },
        index=["plant_a", "plant_b", "plant_c"],
    )


def test_features_are_rowwise_and_do_not_mutate_inputs() -> None:
    frame = example()
    before = frame.copy(deep=True)
    actual = engineer_features(frame)
    pd.testing.assert_frame_equal(before, frame)
    assert actual.loc["plant_a", "capacity_per_worker"] == 2.0
    assert actual.loc["plant_a", "eaf_share"] == 0.4
    assert pd.isna(actual.loc["plant_b", "capacity_per_worker"])
    assert pd.isna(actual.loc["plant_b", "eaf_share"])
    assert pd.isna(actual.loc["plant_c", "log_steel_capacity"])
    pd.testing.assert_frame_equal(
        actual.loc[["plant_a"]], engineer_features(frame.loc[["plant_a"]])
    )


@pytest.mark.parametrize("issue", ["negative", "missing", "extra", "status"])
def test_schema_rejects_bad_inputs(issue: str) -> None:
    frame = example()
    if issue == "negative":
        frame.loc["plant_a", "steel_capacity_ttpa"] = -1.0
    elif issue == "missing":
        frame = frame.drop(columns="workforce")
    elif issue == "extra":
        frame["production_2024_ttpa"] = 1000.0
    else:
        frame.loc["plant_a", "active_status"] = "typo"
    with pytest.raises(pa.errors.SchemaErrors):
        input_schema().validate(frame, lazy=True)


def test_schema_accepts_unknown_measurements_and_new_country() -> None:
    validated = input_schema().validate(example(), lazy=True)
    assert validated.shape == (3, len(FEATURES))


def test_stored_pipeline_matches_notebook_and_serving() -> None:
    artifacts = BASE / "artifacts"
    data = pd.read_csv(artifacts / "clean_plant_data.csv", index_col="GEM plant ID")
    expected = pd.read_csv(artifacts / "test_predictions.csv", index_col="GEM plant ID")
    frame = data.loc[expected.index, FEATURES]
    actual = predict_batch(frame, artifacts)
    np.testing.assert_allclose(
        actual, expected["predicted_ttpa"], rtol=1e-10, atol=1e-8
    )
    pipeline = joblib.load(artifacts / "best_pipeline.joblib")
    np.testing.assert_allclose(actual, pipeline.predict(frame), rtol=0, atol=1e-10)

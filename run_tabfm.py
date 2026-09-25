"""Execute one fixed TabFM experiment against the existing lab holdout."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch

FEATURES = [
    "steel_capacity_ttpa",
    "iron_capacity_ttpa",
    "eaf_capacity_ttpa",
    "workforce",
    "plant_age",
    "country",
    "equipment",
    "active_status",
    "log_steel_capacity",
    "capacity_per_worker",
    "eaf_share",
]
SOURCE_REVISION = "fbb665569425fd2f490c6576b3af967876fe11ff"
WEIGHTS_REVISION = "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"
WEIGHTS_SHA256 = "bd5a615b0322a8f04a895038de6df6fbd71430eca750e1d792f31048654674a9"
CONFIG_GIT_BLOB = "61e3f1b7e6042b89fcfb21ef8e866c32b5f43cde"


def sha256(path: Path) -> str:
    """Hash a file without allocating its contents at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Load pretrained weights, predict once, and persist measured evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--cache", type=Path, default=Path(__file__).resolve().parent / ".tabfm-cache"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = Path(
            snapshot_download(
                repo_id="google/tabfm-1.0.0-pytorch",
                revision=WEIGHTS_REVISION,
                allow_patterns=["regression/*", "LICENSE*"],
                cache_dir=str(args.cache),
            )
        )
    checkpoint_dir = args.checkpoint / "regression"
    if not checkpoint_dir.is_dir():
        checkpoint_dir = args.checkpoint
    print("Verifying pinned checkpoint digest...", flush=True)
    checkpoint_sha256 = sha256(checkpoint_dir / "model.safetensors")
    if checkpoint_sha256 != WEIGHTS_SHA256:
        raise ValueError(
            "Checkpoint does not match the pinned official regression weights."
        )
    config_bytes = (checkpoint_dir / "config.json").read_bytes()
    config_blob = hashlib.sha1(
        f"blob {len(config_bytes)}\0".encode() + config_bytes
    ).hexdigest()
    if config_blob != CONFIG_GIT_BLOB:
        raise ValueError(
            "Configuration does not match the pinned official regression model."
        )
    started = time.perf_counter()
    data_path = args.artifacts / "clean_plant_data.csv"
    split_path = args.artifacts / "split_manifest.csv"
    data = pd.read_csv(data_path).set_index("GEM plant ID")
    data["log_steel_capacity"] = np.log1p(data["steel_capacity_ttpa"])
    data["capacity_per_worker"] = data["steel_capacity_ttpa"] / data["workforce"].where(
        data["workforce"] > 0
    )
    data["eaf_share"] = data["eaf_capacity_ttpa"] / data["steel_capacity_ttpa"].where(
        data["steel_capacity_ttpa"] > 0
    )
    data = data.replace([np.inf, -np.inf], np.nan)
    split = pd.read_csv(split_path).set_index("GEM plant ID")
    assert data.index.is_unique and split.index.is_unique
    assert set(data.index) == set(split.index)
    train_ids = split.index[split["split"].eq("train")]
    test_ids = split.index[split["split"].eq("test")]
    assert len(train_ids) == 205 and len(test_ids) == 52
    assert set(train_ids).isdisjoint(test_ids)
    x_train = data.loc[train_ids, FEATURES]
    y_train = data.loc[train_ids, "production_2024_ttpa"].to_numpy()
    x_test = data.loc[test_ids, FEATURES]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    print("Loading pinned regression checkpoint...", flush=True)
    load_started = time.perf_counter()
    model = tabfm_v1_0_0_pytorch.load(
        model_type="regression",
        checkpoint_path=str(checkpoint_dir),
        device="cpu",
        dtype=torch.bfloat16,
    )
    load_seconds = time.perf_counter() - load_started
    regressor = TabFMRegressor(
        model=model,
        n_estimators=1,
        max_num_rows=100,
        random_state=42,
        batch_size=1,
    )
    print("Preparing training context...", flush=True)
    fit_started = time.perf_counter()
    regressor.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_started
    context_ids = {
        method: [
            [str(train_ids[int(index)]) for index in indices] for indices in patterns
        ]
        for method, patterns in regressor.ensemble_generator_.row_subsample_patterns_.items()
    }
    print("Predicting 52 held-out rows...", flush=True)
    predict_started = time.perf_counter()
    predictions = np.asarray(regressor.predict(x_test), dtype=float)
    predict_seconds = time.perf_counter() - predict_started
    assert predictions.shape == (52,) and np.isfinite(predictions).all()
    y_test = data.loc[test_ids, "production_2024_ttpa"].to_numpy()
    result = {
        "status": "completed",
        "source_revision": SOURCE_REVISION,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "weights_repo": "google/tabfm-1.0.0-pytorch",
        "weights_revision": WEIGHTS_REVISION,
        "weights_subfolder": "regression",
        "weights_license": "tabfm-non-commercial-v1.0",
        "checkpoint_sha256": checkpoint_sha256,
        "config_git_blob": config_blob,
        "dataset_sha256": sha256(data_path),
        "split_sha256": sha256(split_path),
        "features": FEATURES,
        "train_rows": 205,
        "test_rows": 52,
        "context_ids": context_ids,
        "config": {
            "max_num_rows": 100,
            "n_estimators": 1,
            "random_state": 42,
            "batch_size": 1,
            "device": "cpu",
            "dtype": "bfloat16",
            "threads": 4,
        },
        "metrics": {
            "rmse": float(root_mean_squared_error(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        },
        "timing_seconds": {
            "load": load_seconds,
            "fit": fit_seconds,
            "predict": predict_seconds,
            "total": time.perf_counter() - started,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "tabfm",
                "torch",
                "numpy",
                "pandas",
                "scikit-learn",
                "huggingface-hub",
                "safetensors",
                "typeguard",
                "jaxtyping",
            )
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "predictions": [
            {
                "GEM plant ID": str(plant_id),
                "actual": float(actual),
                "prediction": float(prediction),
            }
            for plant_id, actual, prediction in zip(
                test_ids, y_test, predictions, strict=True
            )
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()

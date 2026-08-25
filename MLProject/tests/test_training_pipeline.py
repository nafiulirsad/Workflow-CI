"""Smoke test for the retraining entry point and the model it produces."""
import os
import sys

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

MLPROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MLPROJECT_DIR)

import modelling  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    X_train = pd.read_csv(os.path.join(modelling.DATA_DIR, "X_train_scaled.csv")).head(2000)
    y_train = pd.read_csv(os.path.join(modelling.DATA_DIR, "y_train.csv")).head(2000).values.ravel()
    X_test = pd.read_csv(os.path.join(modelling.DATA_DIR, "X_test_scaled.csv")).head(500)
    y_test = pd.read_csv(os.path.join(modelling.DATA_DIR, "y_test.csv")).head(500).values.ravel()
    return X_train, y_train, X_test, y_test


def test_mlproject_file_declares_main_entry_point():
    with open(os.path.join(MLPROJECT_DIR, "MLProject")) as fh:
        content = fh.read()
    assert "entry_points:" in content
    assert "main:" in content
    assert "modelling.py" in content


def test_conda_env_pins_required_dependencies():
    with open(os.path.join(MLPROJECT_DIR, "conda.yaml")) as fh:
        content = fh.read()
    for package in ("mlflow", "scikit-learn", "pandas", "numpy"):
        assert package in content, f"{package} missing from conda.yaml"


def test_resolve_tracking_uri_prefers_env_var(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example-tracking:5000")
    assert modelling.resolve_tracking_uri() == "http://example-tracking:5000"


def test_resolve_tracking_uri_falls_back_to_local_store(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    uri = modelling.resolve_tracking_uri()
    assert uri.startswith("file://")
    assert os.path.isdir(tmp_path / "mlruns")


def test_load_dataset_returns_aligned_arrays():
    X_train, X_test, y_train, y_test = modelling.load_dataset()
    assert X_train.shape[1] == 6
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_model_trains_and_reaches_minimum_quality(sample):
    X_train, y_train, X_test, y_test = sample
    model = RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape
    assert r2_score(y_train, model.predict(X_train)) > 0.9, "model underfits the training data"


def test_model_prediction_is_a_plausible_price(sample):
    X_train, y_train, _, _ = sample
    model = RandomForestRegressor(n_estimators=25, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    prediction = float(model.predict(X_train.head(1))[0])
    assert prediction > 0, "predicted Bitcoin price must be positive"

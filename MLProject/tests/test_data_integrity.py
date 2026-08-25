"""Data quality gate — runs before any retraining happens in CI."""
import os

import numpy as np
import pandas as pd
import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "namadataset_preprocessing")
EXPECTED_FEATURES = ["Close", "Volume USDT", "RSI", "MACD_Hist", "ATR", "KAMAO"]


@pytest.fixture(scope="module")
def dataset():
    return (
        pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv")),
        pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv")),
        pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")),
        pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")),
    )


def test_all_preprocessed_files_exist():
    for name in ("X_train_scaled.csv", "X_test_scaled.csv", "y_train.csv", "y_test.csv"):
        assert os.path.isfile(os.path.join(DATA_DIR, name)), f"missing {name}"


def test_feature_columns_match_contract(dataset):
    X_train, X_test, _, _ = dataset
    assert list(X_train.columns) == EXPECTED_FEATURES
    assert list(X_test.columns) == EXPECTED_FEATURES


def test_features_and_target_have_same_length(dataset):
    X_train, X_test, y_train, y_test = dataset
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_no_missing_or_infinite_values(dataset):
    for frame in dataset:
        assert not frame.isnull().values.any(), "dataset contains NaN"
        assert np.isfinite(frame.to_numpy()).all(), "dataset contains inf"


def test_train_split_is_larger_than_test_split(dataset):
    X_train, X_test, _, _ = dataset
    assert len(X_train) > len(X_test)
    ratio = len(X_test) / (len(X_train) + len(X_test))
    assert 0.15 <= ratio <= 0.25, f"unexpected test ratio {ratio:.3f}"


def test_features_are_standardised(dataset):
    X_train, _, _, _ = dataset
    means = X_train.mean().abs()
    stds = X_train.std()
    assert (means < 0.1).all(), "scaled train features should be centred around 0"
    assert ((stds > 0.5) & (stds < 1.5)).all(), "scaled train features should have unit variance"


def test_target_values_are_positive_prices(dataset):
    _, _, y_train, y_test = dataset
    assert (y_train.iloc[:, 0] > 0).all()
    assert (y_test.iloc[:, 0] > 0).all()

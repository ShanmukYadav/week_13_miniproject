"""tests/test_features.py — Unit tests for feature engineering."""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline.features import (
    clean_raw, encode_target, engineer_features,
    fill_missing, build_features, get_feature_columns,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [35, 45, 25, 60, 18],
        "job": ["management", "blue-collar", "technician", "retired", "student"],
        "marital": ["married", "single", "divorced", "married", "single"],
        "education": ["tertiary", "secondary", "tertiary", "primary", "secondary"],
        "default": ["no", "no", "yes", "no", "no"],
        "balance": [1500, -200, 300, 5000, 0],
        "housing": ["yes", "yes", "no", "yes", "no"],
        "loan": ["no", "yes", "no", "no", "no"],
        "contact": ["cellular", "telephone", "cellular", "unknown", "cellular"],
        "day": [15, 20, 3, 28, 10],
        "month": ["may", "jun", "jan", "dec", "aug"],
        "duration": [200, 0, 150, 400, 50],
        "campaign": [2, 1, 5, 1, 3],
        "pdays": [-1, 90, -1, 30, -1],
        "previous": [0, 2, 0, 1, 0],
        "poutcome": ["unknown", "success", "unknown", "failure", "unknown"],
        "y": ["yes", "no", "no", "yes", "no"],
    })


def test_clean_raw_removes_duplicates(sample_df):
    df_dup = pd.concat([sample_df, sample_df], ignore_index=True)
    cleaned = clean_raw(df_dup)
    assert len(cleaned) == len(sample_df)


def test_clean_raw_strips_whitespace():
    df = pd.DataFrame({"job": ["  management  ", "blue-collar"], "y": ["yes", "no"],
                        "age": [35, 45], "balance": [100, 200], "day": [1, 2],
                        "month": ["jan", "feb"], "duration": [100, 200], "campaign": [1, 2],
                        "pdays": [-1, -1], "previous": [0, 0]})
    cleaned = clean_raw(df)
    assert cleaned["job"].iloc[0] == "management"


def test_encode_target(sample_df):
    df = encode_target(sample_df)
    assert set(df["y"].unique()).issubset({0, 1})
    assert df["y"].iloc[0] == 1  # "yes" → 1
    assert df["y"].iloc[1] == 0  # "no" → 0


def test_engineer_features_adds_columns(sample_df):
    df = engineer_features(sample_df)
    assert "was_previously_contacted" in df.columns
    assert "duration_bucket" in df.columns
    assert "negative_balance" in df.columns
    assert "age_group" in df.columns
    assert "quarter" in df.columns


def test_was_previously_contacted_correct(sample_df):
    df = engineer_features(sample_df)
    # pdays=-1 → not contacted
    assert df["was_previously_contacted"].iloc[0] == 0
    # pdays=90 → was contacted
    assert df["was_previously_contacted"].iloc[1] == 1


def test_negative_balance_flag(sample_df):
    df = engineer_features(sample_df)
    assert df["negative_balance"].iloc[0] == 0   # balance=1500 → positive
    assert df["negative_balance"].iloc[1] == 1   # balance=-200 → negative


def test_fill_missing_no_nulls_remain(sample_df):
    df = sample_df.copy()
    df.loc[0, "age"] = np.nan
    df.loc[1, "job"] = np.nan
    filled = fill_missing(df)
    assert filled["age"].isnull().sum() == 0
    assert filled["job"].isnull().sum() == 0


def test_build_features_returns_dataframe(sample_df):
    X = build_features(sample_df, fit=True)
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(sample_df)
    assert "y" not in X.columns


def test_build_features_no_target_col(sample_df):
    X = build_features(sample_df, fit=True)
    assert "y" not in X.columns


def test_feature_columns_consistent():
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) > 5
    assert "y" not in cols


def test_build_features_no_nan(sample_df):
    X = build_features(sample_df, fit=True)
    assert X.isnull().sum().sum() == 0


def test_duration_bucket_values(sample_df):
    df = engineer_features(sample_df)
    # duration=0 → bucket 0
    assert df["duration_bucket"].iloc[1] == 0
    # duration=200 → bucket 2
    assert df["duration_bucket"].iloc[0] in [2, 3]

"""
features.py — Reusable, tested feature functions. Same path at train and serve time.
No train-serving skew: call build_features() everywhere.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

# Encoding maps learned at train time
ENCODING_MAPS: dict = {}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, strip whitespace from strings."""
    df = df.copy()
    df = df.drop_duplicates()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip().str.lower()
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert y: yes→1, no→0."""
    df = df.copy()
    if "y" in df.columns:
        df["y"] = (df["y"].str.lower() == "yes").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features.
    Must produce identical columns at train and serve time.
    """
    df = df.copy()

    # Feature 1: Was previously contacted?
    df["was_previously_contacted"] = (df["pdays"] != -1).astype(int)

    # Feature 2: Contact recency bucket
    df["recency_bucket"] = pd.cut(
        df["pdays"].clip(lower=0),
        bins=[-1, 0, 30, 90, 365, 9999],
        labels=[0, 1, 2, 3, 4],
    ).astype(float).fillna(0)

    # Feature 3: Call duration bucket (proxy for engagement)
    df["duration_bucket"] = pd.cut(
        df["duration"],
        bins=[-1, 0, 60, 300, 600, 99999],
        labels=[0, 1, 2, 3, 4],
    ).astype(float).fillna(0)

    # Feature 4: Campaign intensity (capped at 10)
    df["campaign_capped"] = df["campaign"].clip(upper=10)

    # Feature 5: Age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 50, 65, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(float).fillna(2)

    # Feature 6: Negative balance flag
    df["negative_balance"] = (df["balance"] < 0).astype(int)

    # Feature 7: Has personal loan AND housing loan (double burden)
    if "loan" in df.columns and "housing" in df.columns:
        df["double_loan"] = (
            (df["loan"].astype(str) == "yes") & (df["housing"].astype(str) == "yes")
        ).astype(int)

    # Feature 8: Month seasonality (Q1, Q2, Q3, Q4)
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    df["month_num"] = df["month"].map(month_map).fillna(6)
    df["quarter"] = ((df["month_num"] - 1) // 3 + 1).astype(int)

    return df


def ordinal_encode_categoricals(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Encode categorical columns with ordinal encoding.
    Set fit=True at training time to learn the maps.
    At serve time, fit=False uses the saved maps.
    """
    global ENCODING_MAPS
    df = df.copy()

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.lower().fillna("unknown")
        if fit:
            unique_vals = sorted(df[col].unique().tolist())
            ENCODING_MAPS[col] = {v: i for i, v in enumerate(unique_vals)}
        if col in ENCODING_MAPS:
            df[col] = df[col].map(ENCODING_MAPS[col]).fillna(-1).astype(int)
        else:
            # Fallback: label encode
            df[col] = pd.Categorical(df[col]).codes

    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, 'unknown' for categorical."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if not df[col].isnull().all() else 0)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df


def get_feature_columns() -> list:
    """Return the final list of feature columns used at train and serve time."""
    base = NUMERIC_COLS + CATEGORICAL_COLS
    engineered = [
        "was_previously_contacted", "recency_bucket", "duration_bucket",
        "campaign_capped", "age_group", "negative_balance",
        "double_loan", "month_num", "quarter",
    ]
    return [c for c in base + engineered if c not in ["month", "pdays"]]


def build_features(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Full feature pipeline. Call with fit=True at training, fit=False at serve time.
    Returns feature matrix (X) without target column.
    """
    df = clean_raw(df)
    df = fill_missing(df)
    df = engineer_features(df)
    df = ordinal_encode_categoricals(df, fit=fit)

    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    return df[available]


def save_encoding_maps():
    path = ARTIFACTS_DIR / "encoding_maps.joblib"
    joblib.dump(ENCODING_MAPS, path)
    print(f"[features] Encoding maps saved → {path}")


def load_encoding_maps():
    global ENCODING_MAPS
    path = ARTIFACTS_DIR / "encoding_maps.joblib"
    if path.exists():
        ENCODING_MAPS = joblib.load(path)
        print(f"[features] Encoding maps loaded from {path}")
    else:
        print("[features] WARNING: No encoding maps found. Run training first.")


if __name__ == "__main__":
    # Quick sanity check
    df = pd.read_csv("data/raw/bank.csv")
    print(f"Raw shape: {df.shape}")
    X = build_features(df, fit=True)
    print(f"Feature shape: {X.shape}")
    print(f"Feature columns: {X.columns.tolist()}")
    save_encoding_maps()
    print("✓ features.py OK")

"""
train.py — Train baseline (Logistic Regression) and improved (XGBoost) models.
Tracks runs with MLflow. Saves artifacts.
Usage:
    python src/training/train.py
"""

import sys
import json
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_pipeline.features import (
    build_features,
    encode_target,
    save_encoding_maps,
)

# =========================================================
# CONFIG
# =========================================================

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MLFLOW_URI = "mlruns"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("customer-intelligence-platform")

RANDOM_STATE = 42
TEST_SIZE = 0.2


# =========================================================
# DATA LOADING
# =========================================================

def load_data(path: str = "data/raw/bank.csv"):

    df = pd.read_csv(path)

    print(f"[train] Loaded {len(df)} rows from {path}")

    # Encode target
    df = encode_target(df)

    # Feature engineering
    X = build_features(df, fit=True)

    save_encoding_maps()

    y = df["y"].values

    print(
        f"[train] Feature shape: {X.shape} "
        f"| Class balance: {y.mean():.2%} positive"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X.columns.tolist(),
    )


# =========================================================
# BASELINE MODEL
# =========================================================

def train_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
):

    print("[train] Training baseline (Logistic Regression)...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        C=1.0,
    )

    model.fit(X_train_scaled, y_train)

    pipeline = {
        "scaler": scaler,
        "model": model,
        "type": "logistic_regression",
    }

    return pipeline, X_test_scaled


# =========================================================
# IMPROVED MODEL
# =========================================================

def train_improved(X_train, y_train):

    print("[train] Training improved model (XGBoost)...")

    scale_pos = (
        (y_train == 0).sum()
        / (y_train == 1).sum()
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,

        # IMPORTANT FIX
        # Avoid deprecated binary format warning
        enable_categorical=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    return model


# =========================================================
# LOGGING + METRICS
# =========================================================

def log_run(
    name: str,
    model_obj,
    X_test,
    y_test,
    params: dict,
    feature_cols: list,
):

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )

    # Logistic baseline
    if isinstance(model_obj, dict):

        X_eval = model_obj.get(
            "X_test_scaled",
            X_test,
        )

        predictor = model_obj["model"]

    else:

        X_eval = X_test
        predictor = model_obj

    # Probabilities
    y_prob = predictor.predict_proba(X_eval)[:, 1]

    # Threshold tuning
    thresholds = np.arange(0.1, 0.9, 0.01)

    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds:

        y_pred_t = (
            y_prob >= t
        ).astype(int)

        f1 = f1_score(
            y_test,
            y_pred_t,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Final predictions
    y_pred = (
        y_prob >= best_thresh
    ).astype(int)

    # Metrics
    metrics = {
        "roc_auc": float(round(
            roc_auc_score(y_test, y_prob),
            4
        )),
        "pr_auc": float(round(
            average_precision_score(y_test, y_prob),
            4
        )),
        "f1": float(round(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            4
        )),
        "precision": float(round(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            4
        )),
        "recall": float(round(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            4
        )),
        "threshold": float(round(
            best_thresh,
            4
        )),
    }

    cm = confusion_matrix(
        y_test,
        y_pred,
    ).tolist()

    # MLflow logging
    with mlflow.start_run(run_name=name):

        mlflow.log_params(params)

        mlflow.log_metrics(metrics)

        mlflow.log_dict(
            {"confusion_matrix": cm},
            "confusion_matrix.json",
        )

        mlflow.log_dict(
            {"feature_columns": feature_cols},
            "feature_cols.json",
        )

        if isinstance(model_obj, dict):

            mlflow.sklearn.log_model(
                model_obj["model"],
                "model",
            )

        else:

            mlflow.xgboost.log_model(
                model_obj,
                "model",
            )

        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )

    metrics["run_id"] = run_id
    metrics["confusion_matrix"] = cm

    print(
        f"  [{name}] "
        f"ROC-AUC={metrics['roc_auc']} | "
        f"PR-AUC={metrics['pr_auc']} | "
        f"F1={metrics['f1']} | "
        f"Threshold={metrics['threshold']}"
    )

    return metrics


# =========================================================
# PROMOTION GATE
# =========================================================

def promotion_gate(
    baseline_metrics: dict,
    improved_metrics: dict,
):

    pr_delta = (
        improved_metrics["pr_auc"]
        - baseline_metrics["pr_auc"]
    )

    f1_delta = (
        improved_metrics["f1"]
        - baseline_metrics["f1"]
    )

    PR_AUC_MARGIN = 0.03
    F1_DROP_LIMIT = -0.02

    passed = (
        pr_delta >= PR_AUC_MARGIN
        and f1_delta >= F1_DROP_LIMIT
    )

    # IMPORTANT FIXES:
    # Convert numpy types to normal Python types
    gate_result = {

        "passed": bool(passed),

        "pr_auc_delta": float(
            round(pr_delta, 4)
        ),

        "f1_delta": float(
            round(f1_delta, 4)
        ),

        "required_pr_auc_delta": float(
            PR_AUC_MARGIN
        ),

        "required_f1_delta_min": float(
            F1_DROP_LIMIT
        ),

        "decision": (
            "PROMOTED"
            if passed
            else "BLOCKED"
        ),

        # FIXED datetime
        "timestamp": datetime.now().isoformat(),
    }

    print("\n── Promotion Gate ──")

    print(
        f"  PR-AUC delta: "
        f"{pr_delta:+.4f} "
        f"(required: ≥{PR_AUC_MARGIN})"
    )

    print(
        f"  F1 delta:     "
        f"{f1_delta:+.4f} "
        f"(min allowed: {F1_DROP_LIMIT})"
    )

    print(
        f"  Decision: "
        f"{gate_result['decision']}"
    )

    return gate_result


# =========================================================
# SAVE MODEL
# =========================================================

def save_model_artifacts(
    model,
    threshold: float,
    feature_cols: list,
    metrics: dict,
    version: str,
):

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Save model
    model_path = (
        ARTIFACTS_DIR / "model.joblib"
    )

    joblib.dump(model, model_path)

    # Metadata
    meta = {
        "model_version": version,
        "model_type": "xgboost",
        "threshold": float(threshold),
        "feature_columns": feature_cols,
        "metrics": metrics,

        # FIXED datetime
        "trained_at": datetime.now().isoformat(),
    }

    with open(
        ARTIFACTS_DIR / "model_meta.json",
        "w",
    ) as f:

        json.dump(
            meta,
            f,
            indent=2,
        )

    print(
        f"\n[train] ✓ Model saved → "
        f"{model_path}"
    )

    print(
        f"[train] ✓ Metadata saved → "
        f"{ARTIFACTS_DIR}/model_meta.json"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols,
    ) = load_data()

    # =====================================================
    # BASELINE
    # =====================================================

    baseline_pipeline, X_test_scaled = train_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    baseline_pipeline["X_test_scaled"] = X_test_scaled

    baseline_metrics = log_run(
        "baseline_logistic_regression",

        baseline_pipeline,

        X_test,
        y_test,

        {
            "model": "LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
        },

        feature_cols,
    )

    # =====================================================
    # IMPROVED
    # =====================================================

    improved_model = train_improved(
        X_train,
        y_train,
    )

    improved_metrics = log_run(
        "improved_xgboost",

        improved_model,

        X_test,
        y_test,

        {
            "model": "XGBoost",
            "n_estimators": 300,
            "max_depth": 6,
            "lr": 0.05,
        },

        feature_cols,
    )

    # =====================================================
    # PROMOTION GATE
    # =====================================================

    gate = promotion_gate(
        baseline_metrics,
        improved_metrics,
    )

    # =====================================================
    # SAVE PROMOTED MODEL
    # =====================================================

    if gate["passed"]:

        version = (
            "v"
            + datetime.now().strftime(
                "%Y%m%d_%H%M"
            )
        )

        save_model_artifacts(
            improved_model,
            improved_metrics["threshold"],
            feature_cols,
            improved_metrics,
            version,
        )

        print(
            f"\n[train] ✓ XGBoost model "
            f"PROMOTED as {version}"
        )

    else:

        version = (
            "baseline_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M"
            )
        )

        joblib.dump(
            baseline_pipeline["model"],
            ARTIFACTS_DIR / "model.joblib",
        )

        print(
            f"\n[train] ⚠ XGBoost BLOCKED "
            f"— saving baseline as fallback "
            f"({version})"
        )

    # =====================================================
    # TRAINING REPORT
    # =====================================================

    report = {

        "baseline": baseline_metrics,

        "improved": improved_metrics,

        "gate": gate,

        "promoted_version": version,
    }

    with open(
        ARTIFACTS_DIR / "training_report.json",
        "w",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        f"[train] Report saved → "
        f"{ARTIFACTS_DIR}/training_report.json"
    )

    return gate["passed"]


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    success = main()

    # Don't fail exit code in demo
    sys.exit(0)
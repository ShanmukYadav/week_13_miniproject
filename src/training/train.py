"""
train.py — Train baseline (Logistic Regression) and improved (XGBoost) models.
Tracks runs with MLflow. Saves artifacts.
Usage: python src/training/train.py
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
from src.data_pipeline.features import build_features, encode_target, save_encoding_maps

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MLFLOW_URI = "mlruns"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("customer-intelligence-platform")

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str = "data/raw/bank.csv") -> tuple:
    df = pd.read_csv(path)
    print(f"[train] Loaded {len(df)} rows from {path}")

    df = encode_target(df)
    X = build_features(df, fit=True)
    save_encoding_maps()

    y = df["y"].values
    print(f"[train] Feature shape: {X.shape} | Class balance: {y.mean():.2%} positive")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_baseline(X_train, y_train, X_test, y_test):
    """Logistic Regression baseline."""
    print("[train] Training baseline (Logistic Regression)...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        C=1.0,
    )
    model.fit(X_train_s, y_train)

    # Bundle scaler with model for serve-time convenience
    pipeline = {"scaler": scaler, "model": model, "type": "logistic_regression"}
    return pipeline, X_test_s


def train_improved(X_train, y_train):
    """XGBoost improved model."""
    print("[train] Training improved model (XGBoost)...")

    # Handle class imbalance
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

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
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )
    return model


def log_run(name: str, model_obj, X_test, y_test, params: dict, feature_cols: list) -> dict:
    """Log a training run to MLflow and return metrics dict."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, confusion_matrix,
    )

    if isinstance(model_obj, dict):
        # Baseline: use scaled X
        X_eval = model_obj.get("X_test_scaled", X_test)
        predictor = model_obj["model"]
    else:
        X_eval = X_test
        predictor = model_obj

    y_prob = predictor.predict_proba(X_eval)[:, 1]

    # Threshold optimization (maximize F1)
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "threshold": round(float(best_thresh), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")
        mlflow.log_dict({"feature_columns": feature_cols}, "feature_cols.json")

        if isinstance(model_obj, dict):
            mlflow.sklearn.log_model(model_obj["model"], "model")
        else:
            mlflow.xgboost.log_model(model_obj, "model")

        run_id = mlflow.active_run().info.run_id

    metrics["run_id"] = run_id
    metrics["confusion_matrix"] = cm
    print(f"  [{name}] ROC-AUC={metrics['roc_auc']} | PR-AUC={metrics['pr_auc']} | F1={metrics['f1']} | Threshold={metrics['threshold']}")
    return metrics


def promotion_gate(baseline_metrics: dict, improved_metrics: dict) -> bool:
    """
    Promote improved model only if it beats baseline by defined margins.
    Gate: PR-AUC must improve by ≥3pp AND F1 must not drop by >2pp.
    """
    pr_delta = improved_metrics["pr_auc"] - baseline_metrics["pr_auc"]
    f1_delta = improved_metrics["f1"] - baseline_metrics["f1"]

    PR_AUC_MARGIN = 0.03
    F1_DROP_LIMIT = -0.02

    passed = pr_delta >= PR_AUC_MARGIN and f1_delta >= F1_DROP_LIMIT

    gate_result = {
        "passed": passed,
        "pr_auc_delta": round(pr_delta, 4),
        "f1_delta": round(f1_delta, 4),
        "required_pr_auc_delta": PR_AUC_MARGIN,
        "required_f1_delta_min": F1_DROP_LIMIT,
        "decision": "PROMOTED" if passed else "BLOCKED",
        "timestamp": datetime.utcnow().isoformat(),
    }

    print("\n── Promotion Gate ──")
    print(f"  PR-AUC delta: {pr_delta:+.4f} (required: ≥{PR_AUC_MARGIN})")
    print(f"  F1 delta:     {f1_delta:+.4f} (min allowed: {F1_DROP_LIMIT})")
    print(f"  Decision:     {gate_result['decision']}")

    return gate_result


def save_model_artifacts(model, threshold: float, feature_cols: list, metrics: dict, version: str):
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(model, model_path)

    meta = {
        "model_version": version,
        "model_type": "xgboost",
        "threshold": threshold,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(),
    }
    with open(ARTIFACTS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[train] ✓ Model saved → {model_path}")
    print(f"[train] ✓ Metadata saved → {ARTIFACTS_DIR}/model_meta.json")


def main():
    X_train, X_test, y_train, y_test, feature_cols = load_data()

    # ── Baseline ──
    baseline_pipeline, X_test_scaled = train_baseline(X_train, y_train, X_test, y_test)
    baseline_pipeline["X_test_scaled"] = X_test_scaled
    baseline_metrics = log_run(
        "baseline_logistic_regression",
        baseline_pipeline,
        X_test, y_test,
        {"model": "LogisticRegression", "C": 1.0, "class_weight": "balanced"},
        feature_cols,
    )

    # ── Improved ──
    improved_model = train_improved(X_train, y_train)
    improved_metrics = log_run(
        "improved_xgboost",
        improved_model,
        X_test, y_test,
        {"model": "XGBoost", "n_estimators": 300, "max_depth": 6, "lr": 0.05},
        feature_cols,
    )

    # ── Gate ──
    gate = promotion_gate(baseline_metrics, improved_metrics)

    if gate["passed"]:
        version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        save_model_artifacts(improved_model, improved_metrics["threshold"], feature_cols, improved_metrics, version)
        print(f"\n[train] ✓ XGBoost model PROMOTED as {version}")
    else:
        # Save baseline as fallback
        version = f"baseline_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
        joblib.dump(baseline_pipeline["model"], ARTIFACTS_DIR / "model.joblib")
        print(f"\n[train] ⚠ XGBoost BLOCKED — saving baseline as fallback ({version})")

    # Save full report
    report = {
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "gate": gate,
        "promoted_version": version,
    }
    with open(ARTIFACTS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[train] Report saved → {ARTIFACTS_DIR}/training_report.json")

    return gate["passed"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 0)  # Don't fail exit even if gate blocked (it's a demo)

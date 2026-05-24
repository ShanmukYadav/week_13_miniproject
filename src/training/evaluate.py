"""
evaluate.py — Generate full evaluation report: ROC-AUC, PR-AUC, F1, calibration, confusion matrix.
Usage: python src/training/evaluate.py
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_pipeline.features import build_features, encode_target, load_encoding_maps

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("docs")
REPORTS_DIR.mkdir(exist_ok=True)


def load_model_and_meta():
    meta_path = ARTIFACTS_DIR / "model_meta.json"
    model_path = ARTIFACTS_DIR / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Run train.py first.")

    model = joblib.load(model_path)
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    return model, meta


def evaluate(data_path: str = "data/raw/bank.csv") -> dict:
    load_encoding_maps()
    model, meta = load_model_and_meta()

    df = pd.read_csv(data_path)
    df = encode_target(df)
    X = build_features(df, fit=False)
    y = df["y"].values

    # Use 20% test split (same split as train.py)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    threshold = meta.get("threshold", 0.5)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "brier_score": round(brier_score_loss(y_test, y_prob), 4),
        "threshold": threshold,
        "n_test": len(y_test),
        "positive_rate_test": round(float(y_test.mean()), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics["confusion_matrix"] = cm

    # Business interpretation
    tn, fp, fn, tp = np.array(cm).ravel()
    metrics["business_reading"] = {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "interpretation": (
            f"At threshold {threshold:.2f}: Model correctly identifies {tp} likely subscribers. "
            f"Wastes {fp} outreach calls on non-subscribers. "
            f"Misses {fn} potential subscribers. "
            f"Precision {metrics['precision']:.0%} means 1 in {round(1/metrics['precision'])} calls converts."
        ),
    }

    # Save plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f"AUC={metrics['roc_auc']}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, label=f"PR-AUC={metrics['pr_auc']}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    axes[2].plot(mean_pred, fraction_pos, "s-", label="Model")
    axes[2].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes[2].set_title(f"Calibration (Brier={metrics['brier_score']})")
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Fraction Positives")
    axes[2].legend()

    plt.tight_layout()
    plot_path = REPORTS_DIR / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=100)
    plt.close()

    # Save report
    report = {
        "model_version": meta.get("model_version", "unknown"),
        "model_type": meta.get("model_type", "unknown"),
        "metrics": metrics,
        "feature_columns": meta.get("feature_columns", []),
    }
    report_path = REPORTS_DIR / "model_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n── Model Evaluation Report ──")
    for k, v in metrics.items():
        if k not in ("confusion_matrix", "business_reading"):
            print(f"  {k}: {v}")
    print(f"\n  Confusion Matrix: {cm}")
    print(f"\n  Business Reading: {metrics['business_reading']['interpretation']}")
    print(f"\n  Report saved → {report_path}")
    print(f"  Plots saved  → {plot_path}")

    return report


if __name__ == "__main__":
    data_path = "data/raw/bank.csv"
    if not Path(data_path).exists():
        data_path = "data/samples/bank_sample.csv"
    evaluate(data_path)

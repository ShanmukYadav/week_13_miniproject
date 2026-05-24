"""
monitoring/ml_drift.py — Generate ML drift report using Evidently on simulated shift.
Usage: python monitoring/ml_drift.py
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline.features import build_features, encode_target, load_encoding_maps

REPORTS_DIR = Path("docs")
REPORTS_DIR.mkdir(exist_ok=True)

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import DatasetDriftMetric, DataDriftTable
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("[drift] Evidently not installed, using custom drift detection.")


def load_reference_data(path: str = "data/raw/bank.csv") -> pd.DataFrame:
    if not Path(path).exists():
        path = "data/samples/bank_sample.csv"
    df = pd.read_csv(path)
    df = encode_target(df)
    X = build_features(df, fit=False)
    X["y"] = df["y"].values
    return X


def simulate_drift(reference: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate production data drift:
    - Age distribution shifts older (campaign targeting seniors)
    - Balance drops (economic downturn scenario)
    - Campaign intensity increases
    - Missing values introduced in some columns
    """
    np.random.seed(99)
    current = reference.copy()
    n = len(current)

    print("[drift] Simulating distribution shift...")

    # Shift 1: Age distribution shifts older
    if "age" in current.columns:
        current["age"] = (current["age"] * 1.15 + np.random.normal(5, 3, n)).clip(18, 100).astype(int)

    # Shift 2: Balance drops significantly
    if "balance" in current.columns:
        current["balance"] = (current["balance"] * 0.6 - 500 + np.random.normal(0, 200, n)).astype(int)

    # Shift 3: Campaign intensity increases
    if "campaign_capped" in current.columns:
        current["campaign_capped"] = (current["campaign_capped"] + 2).clip(1, 10)
    if "campaign" in current.columns:
        current["campaign"] = (current["campaign"] + 2).clip(1, 20)

    # Shift 4: Duration drops (shorter calls)
    if "duration" in current.columns:
        current["duration"] = (current["duration"] * 0.7).astype(int)
    if "duration_bucket" in current.columns:
        current["duration_bucket"] = (current["duration_bucket"] - 1).clip(0, 4)

    # Shift 5: Target drift (lower conversion rate)
    if "y" in current.columns:
        # Flip some positives to negatives
        pos_mask = current["y"] == 1
        flip_n = int(pos_mask.sum() * 0.4)
        flip_idx = current[pos_mask].sample(flip_n, random_state=99).index
        current.loc[flip_idx, "y"] = 0

    print(f"[drift] Reference positive rate: {reference['y'].mean():.2%}")
    print(f"[drift] Current positive rate:   {current['y'].mean():.2%}")

    return current


def compute_psi(reference_col: pd.Series, current_col: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index."""
    min_val = min(reference_col.min(), current_col.min())
    max_val = max(reference_col.max(), current_col.max())
    bins = np.linspace(min_val, max_val + 1e-6, n_bins + 1)

    ref_pct = np.histogram(reference_col, bins=bins)[0] / len(reference_col)
    cur_pct = np.histogram(current_col, bins=bins)[0] / len(current_col)

    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def run_custom_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Custom drift detection when Evidently is unavailable."""
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    if "y" in numeric_cols:
        numeric_cols.remove("y")

    results = {}
    drifted_cols = []

    for col in numeric_cols[:15]:  # Top 15 features
        if col not in current.columns:
            continue
        psi = compute_psi(reference[col], current[col])
        # KS statistic
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(reference[col].dropna(), current[col].dropna())

        # PSI thresholds: <0.1 stable, 0.1-0.2 slight, >0.2 significant
        drift_level = "stable" if psi < 0.1 else ("slight" if psi < 0.2 else "significant")
        drifted = psi >= 0.2 or ks_p < 0.05

        results[col] = {
            "psi": round(psi, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 6),
            "drift_level": drift_level,
            "drifted": bool(drifted),
        }
        if drifted:
            drifted_cols.append(col)

    return {"column_drift": results, "drifted_columns": drifted_cols}


def generate_drift_plots(reference: pd.DataFrame, current: pd.DataFrame, drift_results: dict):
    """Plot distributions for drifted columns."""
    drifted_cols = drift_results.get("drifted_columns", [])[:4]
    if not drifted_cols:
        return

    fig, axes = plt.subplots(2, min(2, len(drifted_cols)), figsize=(12, 8))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(drifted_cols[:4]):
        if i >= len(axes):
            break
        axes[i].hist(reference[col].dropna(), bins=30, alpha=0.5, label="Reference", color="blue")
        axes[i].hist(current[col].dropna(), bins=30, alpha=0.5, label="Current", color="red")
        psi = drift_results["column_drift"].get(col, {}).get("psi", 0)
        axes[i].set_title(f"{col}\nPSI={psi:.3f}")
        axes[i].legend()
        axes[i].set_xlabel(col)

    plt.tight_layout()
    plot_path = REPORTS_DIR / "drift_plots.png"
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"[drift] Plots saved → {plot_path}")


def main():
    print("[drift] Loading reference data...")
    load_encoding_maps()

    ref_path = "data/raw/bank.csv"
    if not Path(ref_path).exists():
        ref_path = "data/samples/bank_sample.csv"

    reference = load_reference_data(ref_path)
    current = simulate_drift(reference)

    print(f"[drift] Reference: {len(reference)} rows | Current: {len(current)} rows")

    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "reference_size": len(reference),
        "current_size": len(current),
        "reference_positive_rate": round(float(reference["y"].mean()), 4),
        "current_positive_rate": round(float(current["y"].mean()), 4),
        "target_drift": {
            "drifted": abs(float(reference["y"].mean()) - float(current["y"].mean())) > 0.05,
            "delta": round(float(current["y"].mean() - reference["y"].mean()), 4),
        },
    }

    if EVIDENTLY_AVAILABLE:
        print("[drift] Generating Evidently report...")
        try:
            ref_feat = reference.drop(columns=["y"])
            cur_feat = current.drop(columns=["y"])

            ev_report = Report(metrics=[DataDriftPreset()])
            ev_report.run(reference_data=ref_feat.head(1000), current_data=cur_feat.head(1000))

            report_path_html = REPORTS_DIR / "drift_report.html"
            ev_report.save_html(str(report_path_html))
            print(f"[drift] Evidently HTML report → {report_path_html}")

            ev_dict = ev_report.as_dict()
            report_data["evidently_summary"] = str(ev_dict)[:1000]
        except Exception as e:
            print(f"[drift] Evidently failed: {e}, falling back to custom.")
            EVIDENTLY_AVAILABLE_LOCAL = False
        else:
            EVIDENTLY_AVAILABLE_LOCAL = True
    else:
        EVIDENTLY_AVAILABLE_LOCAL = False

    # Custom drift always runs (for JSON report)
    print("[drift] Computing PSI and KS statistics...")
    drift_results = run_custom_drift_report(reference, current)
    report_data.update(drift_results)

    n_drifted = len(drift_results.get("drifted_columns", []))
    n_total = len(drift_results.get("column_drift", {}))
    report_data["drift_summary"] = {
        "total_features_checked": n_total,
        "drifted_features": n_drifted,
        "drift_rate": round(n_drifted / max(1, n_total), 4),
        "recommendation": (
            "RETRAIN" if n_drifted > n_total * 0.3 or report_data["target_drift"]["drifted"]
            else "MONITOR"
        ),
    }

    # Print summary
    print("\n── Drift Report Summary ──")
    print(f"  Target drift: {report_data['target_drift']}")
    print(f"  Drifted features: {n_drifted}/{n_total}")
    print(f"  Drifted columns: {drift_results.get('drifted_columns', [])}")
    print(f"  Recommendation: {report_data['drift_summary']['recommendation']}")

    generate_drift_plots(reference, current, drift_results)

    # Save JSON report
    report_path = REPORTS_DIR / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"\n[drift] ✓ Report saved → {report_path}")

    return report_data


if __name__ == "__main__":
    main()

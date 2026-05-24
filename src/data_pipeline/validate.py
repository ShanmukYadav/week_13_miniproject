"""
validate.py — Schema, missing-value, and business-rule validation.
Usage: python src/data_pipeline/validate.py
"""
import sys
import json
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from pathlib import Path
from datetime import datetime

RAW_DIR = Path("data/raw")
REPORT_PATH = Path("data/validation_report.json")

# ─────────────────────────────────────────────
# Bank Marketing Schema
# ─────────────────────────────────────────────
BANK_SCHEMA = DataFrameSchema(
    {
        "age": Column(int, Check.in_range(18, 100), nullable=False),
        "job": Column(str, nullable=True),
        "marital": Column(str, Check.isin(["married", "single", "divorced", "unknown"]), nullable=True),
        "education": Column(str, nullable=True),
        "default": Column(str, Check.isin(["yes", "no", "unknown"]), nullable=True),
        "balance": Column(int, nullable=False),
        "housing": Column(str, Check.isin(["yes", "no", "unknown"]), nullable=True),
        "loan": Column(str, Check.isin(["yes", "no", "unknown"]), nullable=True),
        "contact": Column(str, nullable=True),
        "day": Column(int, Check.in_range(1, 31), nullable=False),
        "month": Column(str, nullable=False),
        "duration": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "campaign": Column(int, Check.greater_than_or_equal_to(1), nullable=False),
        "pdays": Column(int, nullable=False),
        "previous": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
        "poutcome": Column(str, nullable=True),
        "y": Column(str, Check.isin(["yes", "no"]), nullable=False),
    },
    coerce=True,
)

# ─────────────────────────────────────────────
# Business Rules (Bank Marketing)
# ─────────────────────────────────────────────
def check_bank_business_rules(df: pd.DataFrame) -> dict:
    results = {}

    # Rule 1: Duration > 0 for all successful contacts
    r1 = df[df["duration"] == 0]["y"].value_counts().get("yes", 0)
    results["rule_1_zero_duration_no_conversion"] = {
        "pass": r1 == 0,
        "detail": f"Contacts with duration=0 and y=yes: {r1} (should be 0)",
    }

    # Rule 2: Campaign contacts must be >= 1
    r2 = (df["campaign"] < 1).sum()
    results["rule_2_campaign_at_least_1"] = {
        "pass": int(r2) == 0,
        "detail": f"Rows with campaign < 1: {r2}",
    }

    # Rule 3: Age must be between 18 and 100
    r3 = ((df["age"] < 18) | (df["age"] > 100)).sum()
    results["rule_3_age_in_range"] = {
        "pass": int(r3) == 0,
        "detail": f"Rows with age outside 18-100: {r3}",
    }

    # Rule 4: pdays = -1 means never contacted; previous should be 0
    r4_mask = (df["pdays"] == -1) & (df["previous"] > 0)
    r4 = r4_mask.sum()
    results["rule_4_pdays_previous_consistency"] = {
        "pass": int(r4) == 0,
        "detail": f"Rows where pdays=-1 but previous>0: {r4}",
    }

    # Rule 5: No null values in target column
    r5 = df["y"].isnull().sum()
    results["rule_5_target_not_null"] = {
        "pass": int(r5) == 0,
        "detail": f"Null values in target y: {r5}",
    }

    # Rule 6: Class balance check (not extreme — at least 5% positive)
    pos_rate = (df["y"] == "yes").mean()
    results["rule_6_class_balance_check"] = {
        "pass": 0.05 <= pos_rate <= 0.95,
        "detail": f"Positive class rate: {pos_rate:.2%} (acceptable: 5%-95%)",
    }

    # Rule 7: Balance field has no extreme outliers suggesting corruption
    balance_max = df["balance"].abs().max()
    results["rule_7_balance_not_corrupted"] = {
        "pass": bool(balance_max < 200_000),
        "detail": f"Max absolute balance: {balance_max} (threshold: 200000)",
    }

    return results


# ─────────────────────────────────────────────
# Complaints Schema (flexible)
# ─────────────────────────────────────────────
def check_complaints(df: pd.DataFrame) -> dict:
    results = {}

    # Detect narrative column
    narrative_col = None
    for col in ["narrative", "complaint_what_happened", "Consumer complaint narrative"]:
        if col in df.columns:
            narrative_col = col
            break

    if narrative_col is None:
        return {"error": "No narrative column found"}

    # Rule 1: Narrative not null
    null_cnt = df[narrative_col].isnull().sum()
    results["complaint_rule_1_narrative_not_null"] = {
        "pass": int(null_cnt) == 0,
        "detail": f"Null narratives: {null_cnt}",
    }

    # Rule 2: Narrative length > 50 chars
    short = (df[narrative_col].astype(str).str.len() < 50).sum()
    results["complaint_rule_2_narrative_min_length"] = {
        "pass": int(short) == 0,
        "detail": f"Narratives < 50 chars: {short}",
    }

    # Rule 3: No duplicate complaint IDs
    if "complaint_id" in df.columns:
        dups = df["complaint_id"].duplicated().sum()
        results["complaint_rule_3_unique_ids"] = {
            "pass": int(dups) == 0,
            "detail": f"Duplicate complaint IDs: {dups}",
        }

    # Rule 4: Product column populated
    if "product" in df.columns:
        null_prod = df["product"].isnull().sum()
        results["complaint_rule_4_product_populated"] = {
            "pass": float(null_prod / len(df)) < 0.1,
            "detail": f"Null product: {null_prod} ({null_prod/len(df):.1%})",
        }

    # Rule 5: Dataset has at least 1000 rows
    results["complaint_rule_5_minimum_rows"] = {
        "pass": len(df) >= 1000,
        "detail": f"Row count: {len(df)} (minimum: 1000)",
    }

    return results


def run_validation(bank_path: Path, complaints_path: Path) -> dict:
    report = {"timestamp": datetime.utcnow().isoformat(), "results": {}}
    all_passed = True

    # ── Bank Marketing ──
    print("[validate] Checking Bank Marketing data...")
    df_bank = pd.read_csv(bank_path)
    print(f"  Loaded {len(df_bank)} rows, {df_bank.shape[1]} columns")

    # Schema validation
    try:
        BANK_SCHEMA.validate(df_bank)
        report["results"]["bank_schema"] = {"pass": True, "detail": "Schema valid"}
    except pa.errors.SchemaErrors as e:
        report["results"]["bank_schema"] = {"pass": False, "detail": str(e.failure_cases.head(5).to_dict())}
        all_passed = False

    # Missing value check
    missing = df_bank.isnull().sum()
    missing_pct = (missing / len(df_bank) * 100).round(2)
    high_missing = missing_pct[missing_pct > 20].to_dict()
    report["results"]["bank_missing_values"] = {
        "pass": len(high_missing) == 0,
        "detail": f"Columns >20% missing: {high_missing}",
    }

    # Business rules
    biz_rules = check_bank_business_rules(df_bank)
    report["results"].update(biz_rules)
    for v in biz_rules.values():
        if not v["pass"]:
            all_passed = False

    # ── Complaints ──
    print("[validate] Checking CFPB Complaints data...")
    df_comp = pd.read_csv(complaints_path)
    print(f"  Loaded {len(df_comp)} rows, {df_comp.shape[1]} columns")
    comp_rules = check_complaints(df_comp)
    report["results"].update(comp_rules)
    for v in comp_rules.values():
        if not v.get("pass", True):
            all_passed = False

    report["overall_pass"] = all_passed

    # Save report
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[validate] Report saved → {REPORT_PATH}")

    # Print summary
    print("\n── Validation Summary ──")
    for k, v in report["results"].items():
        status = "✓ PASS" if v.get("pass") else "✗ FAIL"
        print(f"  {status} | {k}: {v.get('detail', '')}")

    print(f"\n[validate] Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return report


if __name__ == "__main__":
    bank_path = Path("data/raw/bank.csv")
    complaints_path = Path("data/raw/complaints.csv")

    # Fallback to samples for CI
    if not bank_path.exists():
        bank_path = Path("data/samples/bank_sample.csv")
    if not complaints_path.exists():
        complaints_path = Path("data/samples/complaints_sample.csv")

    if not bank_path.exists() or not complaints_path.exists():
        print("[validate] ERROR: Run ingest.py first.")
        sys.exit(1)

    report = run_validation(bank_path, complaints_path)
    if not report["overall_pass"]:
        sys.exit(1)

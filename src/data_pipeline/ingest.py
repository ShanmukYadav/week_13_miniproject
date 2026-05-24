"""
ingest.py — Fast download version
Usage:
    python src/data_pipeline/ingest.py
"""

import os
import io
import zipfile
import hashlib
import json
import requests
import pandas as pd

from pathlib import Path
from datetime import datetime

RAW_DIR = Path("data/raw")
SAMPLE_DIR = Path("data/samples")

RAW_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

MANIFEST_PATH = Path("data/raw/manifest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


# =========================================================
# BANK MARKETING DATASET
# =========================================================

def download_bank_marketing() -> Path:

    out_path = RAW_DIR / "bank.csv"

    if out_path.exists():
        print(f"[ingest] Bank dataset already exists → {out_path}")
        return out_path

    print("[ingest] Downloading UCI Bank Marketing dataset...")

    resp = requests.get(BANK_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:

        names = z.namelist()
        print(f"[ingest] Zip contents: {names}")

        # Find nested zip
        inner_zip_name = next(
            (n for n in names if n.endswith(".zip")),
            None
        )

        if inner_zip_name is None:
            raise Exception("Nested zip not found")

        with z.open(inner_zip_name) as inner_file:

            with zipfile.ZipFile(io.BytesIO(inner_file.read())) as inner_z:

                inner_names = inner_z.namelist()
                print(f"[ingest] Inner zip contents: {inner_names}")

                target = next(
                    (n for n in inner_names if "bank-full" in n),
                    None
                )

                if target is None:
                    raise Exception("bank-full.csv not found")

                with inner_z.open(target) as f:
                    df = pd.read_csv(
                        f,
                        sep=";",
                        encoding="latin-1"
                    )

    df.to_csv(out_path, index=False)

    print(f"[ingest] Saved {len(df)} rows → {out_path}")

    # Small CI sample
    sample = (
        df.groupby("y", group_keys=False)
        .apply(
            lambda x: x.sample(
                min(100, len(x)),
                random_state=42
            ),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    sample_path = SAMPLE_DIR / "bank_sample.csv"

    sample.to_csv(sample_path, index=False)

    print(f"[ingest] CI sample saved ({len(sample)} rows) → {sample_path}")

    return out_path


# =========================================================
# CFPB COMPLAINTS DATASET (FAST VERSION)
# =========================================================

def download_cfpb_complaints(n: int = 5000) -> Path:

    out_path = RAW_DIR / "complaints.csv"

    if out_path.exists():
        print(f"[ingest] CFPB dataset already exists → {out_path}")
        return out_path

    print(f"[ingest] Downloading CFPB complaints (n={n})...")

    try:

        # FAST API VERSION
        api_url = (
            "https://www.consumerfinance.gov/data-research/"
            "consumer-complaints/search/api/v1/"
            f"?size={n}"
        )

        print("[ingest] Using fast CFPB API...")

        resp = requests.get(api_url, timeout=60)

        resp.raise_for_status()

        data = resp.json()

        # API structure handling
        if "hits" in data:
            records = data["hits"]
        else:
            records = data

        df = pd.DataFrame(records)

        # Standardize columns
        col_map = {
            "consumer_complaint_narrative": "narrative",
            "product": "product",
            "issue": "issue",
            "company": "company",
            "company_response": "company_response",
            "date_received": "date_received",
            "state": "state",
            "complaint_id": "complaint_id",
        }

        df = df.rename(columns=col_map)

        # Filter narratives
        if "narrative" in df.columns:

            df = df.dropna(subset=["narrative"])

            df = df[
                df["narrative"]
                .astype(str)
                .str.len() > 50
            ]

        # Keep only required rows
        df = df.head(n).reset_index(drop=True)

        print(f"[ingest] Downloaded {len(df)} complaint records")

    except Exception as e:

        print(f"[ingest] API download failed: {e}")

        print("[ingest] Falling back to synthetic data...")

        df = _generate_synthetic_complaints(
            n=min(n, 1000)
        )

    # Save full dataset
    df.to_csv(out_path, index=False)

    print(f"[ingest] Saved {len(df)} complaints → {out_path}")

    # Sample dataset
    sample = df.sample(
        min(200, len(df)),
        random_state=42
    )

    sample_path = SAMPLE_DIR / "complaints_sample.csv"

    sample.to_csv(sample_path, index=False)

    print(
        f"[ingest] CI sample saved "
        f"({len(sample)} rows) → {sample_path}"
    )

    return out_path


# =========================================================
# SYNTHETIC FALLBACK
# =========================================================

def _generate_synthetic_complaints(n: int = 500):

    import random

    random.seed(42)

    products = [
        "Mortgage",
        "Credit card",
        "Checking account",
        "Student loan",
        "Personal loan"
    ]

    issues = [
        "Incorrect information",
        "Unauthorized charges",
        "Managing account",
        "Payment issue",
        "Loan processing"
    ]

    companies = [
        "Bank of America",
        "Wells Fargo",
        "Chase",
        "Citibank",
        "Capital One"
    ]

    narratives = [
        "I have been trying to resolve this issue for months.",
        "They charged me unauthorized fees repeatedly.",
        "Incorrect information appears on my credit report.",
    ]

    responses = [
        "Closed with explanation",
        "Closed with monetary relief"
    ]

    states = ["CA", "TX", "FL", "NY", "PA"]

    rows = []

    for i in range(n):

        rows.append({
            "complaint_id": f"SYNTH-{i+1:05d}",
            "product": random.choice(products),
            "issue": random.choice(issues),
            "company": random.choice(companies),
            "narrative": (
                random.choice(narratives)
                + f" Case number {i+1}."
            ),
            "company_response": random.choice(responses),
            "date_received": (
                f"2024-{random.randint(1,12):02d}-"
                f"{random.randint(1,28):02d}"
            ),
            "state": random.choice(states),
        })

    print(f"[ingest] Generated {n} synthetic complaints")

    return pd.DataFrame(rows)


# =========================================================
# MANIFEST
# =========================================================

def save_manifest(paths: dict):

    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "files": {}
    }

    for name, path in paths.items():

        p = Path(path)

        if p.exists():

            manifest["files"][name] = {
                "path": str(p),
                "rows": len(pd.read_csv(p)),
                "sha256": sha256(p),
                "size_mb": round(
                    p.stat().st_size / 1e6,
                    2
                ),
            }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ingest] Manifest saved → {MANIFEST_PATH}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    bank_path = download_bank_marketing()

    cfpb_path = download_cfpb_complaints(n=5000)

    save_manifest({
        "bank": bank_path,
        "complaints": cfpb_path
    })

    print("\n[ingest] ✓ All data downloaded successfully.")
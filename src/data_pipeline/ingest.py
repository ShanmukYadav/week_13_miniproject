"""
ingest.py — Download UCI Bank Marketing + CFPB complaint sample.
Usage: python src/data_pipeline/ingest.py
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
CFPB_API = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
    "?size=5000&field=all&format=json&has_narrative=true"
    "&product=Credit%20card&product=Checking%20or%20savings%20account"
    "&product=Mortgage&product=Student%20loan"
)

MANIFEST_PATH = Path("data/raw/manifest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_bank_marketing() -> Path:
    out_path = RAW_DIR / "bank.csv"
    if out_path.exists():
        print(f"[ingest] Bank Marketing already exists at {out_path}")
        return out_path

    print("[ingest] Downloading UCI Bank Marketing dataset...")
    resp = requests.get(BANK_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        # The zip contains bank-full.csv (semicolon-separated)
        names = z.namelist()
        print(f"[ingest] Zip contents: {names}")
        target = next((n for n in names if "bank-full" in n), None)
        if target is None:
            target = next((n for n in names if n.endswith(".csv")), names[0])
        with z.open(target) as f:
            df = pd.read_csv(f, sep=";")

    df.to_csv(out_path, index=False)
    print(f"[ingest] Saved {len(df)} rows → {out_path}")

    # CI sample (200 rows, stratified)
    sample = df.groupby("y", group_keys=False).apply(lambda x: x.sample(min(100, len(x)), random_state=42))
    sample.to_csv(SAMPLE_DIR / "bank_sample.csv", index=False)
    print(f"[ingest] CI sample saved ({len(sample)} rows) → {SAMPLE_DIR}/bank_sample.csv")
    return out_path


def download_cfpb_complaints(n: int = 5000) -> Path:
    out_path = RAW_DIR / "complaints.csv"
    if out_path.exists():
        print(f"[ingest] CFPB complaints already exist at {out_path}")
        return out_path

    print(f"[ingest] Downloading CFPB complaints (n={n})...")
    url = (
        "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
        f"?size={n}&field=all&format=json&has_narrative=true"
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        # Fallback: try the direct CSV download (smaller)
        print("[ingest] API returned no hits, trying CSV fallback...")
        csv_url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
        r2 = requests.get(csv_url, timeout=180, stream=True)
        r2.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r2.content)) as z:
            csv_name = [x for x in z.namelist() if x.endswith(".csv")][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, nrows=n)
    else:
        records = []
        for hit in hits:
            src = hit.get("_source", {})
            records.append({
                "complaint_id": src.get("complaint_id", ""),
                "product": src.get("product", ""),
                "issue": src.get("issue", ""),
                "company": src.get("company", ""),
                "narrative": src.get("complaint_what_happened", ""),
                "company_response": src.get("company_response", ""),
                "date_received": src.get("date_received", ""),
                "state": src.get("state", ""),
            })
        df = pd.DataFrame(records)

    # Keep only rows with a narrative
    df = df.dropna(subset=["narrative"] if "narrative" in df.columns else [df.columns[-1]])
    df = df[df["narrative"].astype(str).str.len() > 50]
    df = df.head(n)

    # Redact obvious PII patterns (XXX placeholders already in CFPB data)
    df.to_csv(out_path, index=False)
    print(f"[ingest] Saved {len(df)} complaints → {out_path}")

    # CI sample
    sample = df.sample(min(200, len(df)), random_state=42)
    sample.to_csv(SAMPLE_DIR / "complaints_sample.csv", index=False)
    print(f"[ingest] CI sample saved ({len(sample)} rows) → {SAMPLE_DIR}/complaints_sample.csv")
    return out_path


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
                "size_mb": round(p.stat().st_size / 1e6, 2),
            }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ingest] Manifest saved → {MANIFEST_PATH}")


if __name__ == "__main__":
    bank_path = download_bank_marketing()
    cfpb_path = download_cfpb_complaints(n=5000)
    save_manifest({"bank": bank_path, "complaints": cfpb_path})
    print("\n[ingest] ✓ All data downloaded successfully.")

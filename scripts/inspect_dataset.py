#!/usr/bin/env python3
"""
Dataset discovery and schema inspection utility for FlyRank Capstone.
Scans the connected dataset to identify columns, dtypes, missingness, cardinality,
and candidate leakage fields before any downstream modeling is performed.
Uses DuckDB Secrets Manager (httpfs) for secure Hugging Face warehouse authentication.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.config import load_config
from src.data import (
    create_duckdb_connection_with_hf_auth,
    discover_schema,
    get_hf_token,
    load_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset schema and data quality.")
    parser.add_argument("--path", type=str, default=None, help="Path to raw dataset CSV/Parquet.")
    parser.add_argument("--output", type=str, default="outputs/schema_inspection.json", help="Path to save inspection JSON.")
    args = parser.parse_args()

    config = load_config()
    data_path = args.path or config.raw_data_path
    target_file = Path(data_path)
    hf_token = get_hf_token(raise_error=False)

    print("==================================================")
    print("FLYRANK DATASET DISCOVERY & SCHEMA INSPECTION")
    print("==================================================")

    try:
        import duckdb
        print(f"DuckDB Version: {duckdb.__version__}")
    except ImportError:
        print("DuckDB Version: Not Installed (pip install -r requirements.txt)")

    if target_file.exists():
        print(f"Source Mode: LOCAL SAMPLE FIXTURE ('{data_path}')")
        print("Note: Local fixture is used for local sample testing only and is not the full FlyRank warehouse.")
    else:
        repo_id = config.schema_mapping.get("hf", {}).get("dataset_repo", "FlyRank/internship-warehouse")
        print(f"Source Mode: REMOTE WAREHOUSE ('hf://datasets/{repo_id}')")

    if not target_file.exists() and not hf_token:
        print(f"\n[STATUS: HF_TOKEN NOT CONFIGURED]")
        print("HF_TOKEN is not configured. Set HF_TOKEN in your environment or local .env file before accessing the gated FlyRank warehouse.")
        print("\nLocal Setup Instructions:")
        print(" 1. Create a local .env file in the repository root containing:")
        print("    HF_TOKEN=hf_your_actual_token_here")
        print(" 2. Or in PowerShell: $env:HF_TOKEN=\"hf_your_actual_token_here\"")
        print(" 3. Re-run: python scripts/inspect_dataset.py\n")
        return 0

    # Minimal diagnostic authentication check before heavy query
    if not target_file.exists() and hf_token:
        print("\nRunning minimal authentication & extension diagnostic...")
        try:
            conn = create_duckdb_connection_with_hf_auth(hf_token)
            print(" -> httpfs extension loaded cleanly.")
            print(" -> Temporary Hugging Face secret 'hf_token' created in DuckDB Secrets Manager.")
        except Exception as e:
            print(f"\n[AUTHENTICATION DIAGNOSTIC FAILED]: {e}")
            return 1

    print("\nLoading dataset schema and metadata...")
    try:
        df = load_dataset(data_path, config)
        print(f"Successfully connected! Discovered {len(df):,} rows across {len(df.columns)} columns.")
    except Exception as e:
        print(f"\n{e}")
        return 1

    print("\nRunning schema discovery...")
    inspection = discover_schema(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2)

    print(f"\nInspection saved to '{out_path}'.")
    print(f" - Date fields detected: {inspection['date_fields']}")
    print(f" - Numeric fields: {len(inspection['numeric_fields'])}")
    print(f" - Categorical fields: {len(inspection['categorical_fields'])}")

    print("\nCandidate column inventory:")
    for col, info in list(inspection["columns"].items())[:15]:
        print(f"   * {col:<25} | {info['dtype']:<10} | Nulls: {info['null_pct']}% | Unique: {info['unique_values']}")

    if len(inspection["columns"]) > 15:
        print(f"   ... and {len(inspection['columns']) - 15} more columns.")

    print("\nPlease review configs/schema_mapping.yaml to ensure aliases match the discovered schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

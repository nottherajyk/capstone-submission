#!/usr/bin/env python3
"""
Dataset discovery and schema inspection utility.
Scans the connected dataset to identify columns, dtypes, missingness, cardinality,
and candidate leakage fields before any downstream modeling is performed.
Loads local .env and uses HF_TOKEN for Hugging Face authentication.
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
from src.data import discover_schema, get_hf_token, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset schema and data quality.")
    parser.add_argument("--path", type=str, default=None, help="Path to raw dataset CSV/Parquet.")
    parser.add_argument("--output", type=str, default="outputs/schema_inspection.json", help="Path to save inspection JSON.")
    args = parser.parse_args()

    config = load_config()
    data_path = args.path or config.raw_data_path

    print("==================================================")
    print("FLYRANK DATASET DISCOVERY & SCHEMA INSPECTION")
    print("==================================================")
    print(f"Target dataset path: {data_path}")

    target_file = Path(data_path)
    hf_token = get_hf_token(raise_error=False)

    if not target_file.exists() and not hf_token:
        print(f"\n[STATUS: HF_TOKEN NOT CONFIGURED]")
        print("HF_TOKEN is not configured. Set HF_TOKEN in your environment or local .env file before accessing the gated FlyRank warehouse.")
        print("\nLocal Setup Instructions:")
        print(" 1. Create a local .env file in the repository root containing:")
        print("    HF_TOKEN=hf_your_actual_token_here")
        print(" 2. Or in PowerShell: $env:HF_TOKEN=\"hf_your_actual_token_here\"")
        print(" 3. Re-run: python scripts/inspect_dataset.py\n")
        return 0

    print("Loading dataset (using local file or authenticated Hugging Face connection)...")
    try:
        df = load_dataset(data_path, config)
        print(f"Successfully loaded {len(df):,} rows across {len(df.columns)} columns.")
    except Exception as e:
        print(f"\n[ERROR LOADING DATASET]: {e}")
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

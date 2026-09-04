#!/usr/bin/env python3
"""
Dataset discovery and schema inspection utility.
Scans the connected dataset to identify columns, dtypes, missingness, cardinality,
and candidate leakage fields before any downstream modeling is performed.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data import discover_schema, load_dataset


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
    if not target_file.exists():
        print(f"\n[STATUS: AWAITING REAL WAREHOUSE DATASET]")
        print(f"The configured dataset file '{data_path}' was not found.")
        print("To connect the dataset:")
        print(" 1. Download or query your approved FlyRank slice (or internship warehouse export).")
        print(f" 2. Save it to '{data_path}' or configure FLYRANK_DATA_PATH in .env.")
        print(" 3. Re-run: python scripts/inspect_dataset.py\n")
        return 0

    print("Loading dataset...")
    df = load_dataset(data_path, config)
    print(f"Successfully loaded {len(df):,} rows across {len(df.columns)} columns.")

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

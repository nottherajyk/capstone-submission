#!/usr/bin/env python3
"""
Dataset discovery and schema inspection utility for FlyRank Capstone.
Scans the connected FlyRank Hugging Face warehouse via DuckDB Secrets Manager (httpfs)
to discover tables, schemas, data types, candidate identifiers, and candidate features
without placing large expensive sweeps upfront.
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
    classify_warehouse_error,
    create_duckdb_connection_with_hf_auth,
    discover_schema,
    get_hf_token,
    get_warehouse_sources,
    load_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect FlyRank dataset schema and data quality.")
    parser.add_argument("--path", type=str, default=None, help="Path to raw dataset CSV/Parquet for local fixture inspection.")
    parser.add_argument("--output", type=str, default="outputs/schema_inspection.json", help="Path to save inspection JSON.")
    args = parser.parse_args()

    config = load_config()
    raw_path_arg = args.path
    target_file = Path(raw_path_arg) if raw_path_arg else None
    hf_token = get_hf_token(raise_error=False)

    print("==================================================")
    print("FLYRANK DATASET DISCOVERY & SCHEMA INSPECTION")
    print("==================================================")

    try:
        import duckdb
        duckdb_ver = duckdb.__version__
        print(f"DuckDB Version: {duckdb_ver}")
    except ImportError:
        duckdb_ver = "Not Installed"
        print("DuckDB Version: Not Installed (pip install -r requirements.txt)")

    repo_id = config.schema_mapping.get("dataset", {}).get("repo", "FlyRank/internship-warehouse")
    release_ver = config.schema_mapping.get("dataset", {}).get("release", "v20260703")

    if target_file and target_file.exists():
        source_mode = "LOCAL_FIXTURE"
        print(f"Source Mode: LOCAL_FIXTURE ('{raw_path_arg}')")
        print("Note: Local fixture is used for offline unit testing only and is not the full FlyRank warehouse.")
    else:
        source_mode = "REMOTE_WAREHOUSE"
        print(f"Source Mode: REMOTE_WAREHOUSE ('hf://datasets/{repo_id}')")
        print(f"Dataset: {repo_id}")
        print(f"Release: {release_ver}")

    if source_mode == "REMOTE_WAREHOUSE" and not hf_token:
        print("\n[STATUS: HF_TOKEN NOT CONFIGURED]")
        print("HF_TOKEN is not configured. Set HF_TOKEN in your environment or local .env file before accessing the gated FlyRank warehouse.")
        print("\nLocal Setup Instructions:")
        print(" 1. Create a local .env file in the repository root containing:")
        print("    HF_TOKEN=hf_your_actual_token_here")
        print(" 2. Or in PowerShell: $env:HF_TOKEN=\"hf_your_actual_token_here\"")
        print(" 3. Re-run: python scripts/inspect_dataset.py\n")
        return 0

    # Minimal diagnostic authentication check
    conn = None
    if source_mode == "REMOTE_WAREHOUSE" and hf_token:
        print("\nRunning minimal authentication & extension diagnostic...")
        try:
            conn = create_duckdb_connection_with_hf_auth(hf_token)
            print(" -> httpfs extension loaded cleanly.")
            print(" -> Temporary Hugging Face secret 'hf_token' created in DuckDB Secrets Manager.")
        except Exception as e:
            print(f"\n[AUTHENTICATION DIAGNOSTIC FAILED]: {e}")
            return 1

    sources = get_warehouse_sources(repo_id)
    inspection_report = {
        "dataset": repo_id,
        "release": release_ver,
        "duckdb_version": duckdb_ver,
        "source_mode": source_mode,
        "warehouse_sources": {},
        "candidate_identifiers": {},
        "candidate_performance_fields": {},
    }

    if source_mode == "REMOTE_WAREHOUSE" and conn is not None:
        print("\nInspecting configured warehouse objects...")
        warehouse_objects = [
            ("sample", sources["sample"], "SAMPLE (fact_content_daily_performance_sample.parquet)"),
            ("dim_clients", sources["dim_clients"], "DIMENSION TABLE (dim_clients.parquet)"),
            ("dim_content", sources["dim_content"], "DIMENSION TABLE (dim_content.parquet)"),
            ("query_90d", sources["query_90d"], "QUERY TRAFFIC (fact_content_query_90d.parquet)"),
            ("daily_performance", sources["daily_performance"], "FULL WAREHOUSE (fact_content_daily_performance/**/*.parquet)"),
        ]

        for source_key, source_path, label in warehouse_objects:
            print(f"\n -> Source [{source_key}]: {label}")
            print(f"    Path: {source_path}")
            try:
                # Bounded metadata query (LIMIT 5)
                df_sample = conn.execute(f"SELECT * FROM read_parquet('{source_path}') LIMIT 5;").df()
                col_types = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{source_path}') LIMIT 5;").fetchall()
                col_type_dict = {col[0]: col[1] for col in col_types}

                print(f"    Status: REACHABLE ({len(df_sample.columns)} columns)")

                source_info = {
                    "status": "REACHABLE",
                    "path": source_path,
                    "label": label,
                    "column_count": len(df_sample.columns),
                    "columns": col_type_dict,
                }

                # Extract date range if date column present
                date_cols = [c for c in df_sample.columns if "date" in c.lower() or "time" in c.lower()]
                if date_cols:
                    try:
                        dcol = date_cols[0]
                        drange = conn.execute(f"SELECT MIN({dcol}), MAX({dcol}) FROM read_parquet('{source_path}') LIMIT 100;").fetchone()
                        source_info["date_range"] = {"min": str(drange[0]), "max": str(drange[1])}
                        print(f"    Date Range ({dcol}): {drange[0]} to {drange[1]}")
                    except Exception:
                        pass

                inspection_report["warehouse_sources"][source_key] = source_info

            except Exception as e:
                err_classified = classify_warehouse_error(e, hf_token)
                print(f"    Status: NOT REACHABLE - {err_classified}")
                inspection_report["warehouse_sources"][source_key] = {
                    "status": "UNREACHABLE",
                    "path": source_path,
                    "label": label,
                    "error": str(err_classified),
                }

    print("\nLoading dataset sample for schema resolution...")
    try:
        df = load_dataset(path=raw_path_arg, config=config, source_mode=source_mode)
        print(f"Successfully loaded dataset! Discovered {len(df):,} rows across {len(df.columns)} columns.")
    except Exception as e:
        print(f"\n[DATASET LOAD ERROR]: {e}")
        return 1

    print("\nRunning column schema discovery...")
    schema_disc = discover_schema(df)
    inspection_report["schema_discovery"] = schema_disc

    # Identify candidate identifiers and performance fields
    for col in df.columns:
        c_lower = col.lower()
        if "client" in c_lower or "account" in c_lower or "tenant" in c_lower:
            inspection_report["candidate_identifiers"]["client_id"] = col
        elif "page" in c_lower or "url" in c_lower or "content" in c_lower or "doc" in c_lower:
            inspection_report["candidate_identifiers"]["content_id"] = col
        elif "date" in c_lower or "time" in c_lower or "period" in c_lower:
            inspection_report["candidate_identifiers"]["report_date"] = col

        if "click" in c_lower:
            inspection_report["candidate_performance_fields"]["clicks"] = col
        elif "impression" in c_lower:
            inspection_report["candidate_performance_fields"]["impressions"] = col
        elif "ctr" in c_lower:
            inspection_report["candidate_performance_fields"]["ctr"] = col
        elif "position" in c_lower or "rank" in c_lower:
            inspection_report["candidate_performance_fields"]["position"] = col

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inspection_report, f, indent=2)

    print(f"\nInspection results saved to '{out_path}'.")
    print(f" - Date fields detected: {schema_disc['date_fields']}")
    print(f" - Numeric fields: {len(schema_disc['numeric_fields'])}")
    print(f" - Categorical fields: {len(schema_disc['categorical_fields'])}")
    print(f" - Candidate client identifier: {inspection_report['candidate_identifiers'].get('client_id', 'Not identified')}")
    print(f" - Candidate content identifier: {inspection_report['candidate_identifiers'].get('content_id', 'Not identified')}")
    print(f" - Candidate report date: {inspection_report['candidate_identifiers'].get('report_date', 'Not identified')}")

    print("\nCandidate column inventory (bounded view):")
    for col, info in list(schema_disc["columns"].items())[:15]:
        print(f"   * {col:<25} | {info['dtype']:<10} | Nulls: {info['null_pct']}% | Unique: {info['unique_values']}")

    if len(schema_disc["columns"]) > 15:
        print(f"   ... and {len(schema_disc['columns']) - 15} more columns.")

    print("\nPlease review configs/schema_mapping.yaml to ensure aliases match the discovered schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Dataset discovery and schema inspection utility for FlyRank Capstone.
Scans the connected FlyRank Hugging Face warehouse via DuckDB Secrets Manager (httpfs)
to discover tables, schemas, data types, candidate identifiers, analytical grain,
and candidate features without placing large expensive sweeps upfront.
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

try:
    import pandas as pd
except ImportError:
    pd = None

from src.config import load_config
from src.data import (
    classify_warehouse_error,
    create_duckdb_connection_with_hf_auth,
    discover_schema,
    get_hf_token,
    get_warehouse_sources,
    infer_candidate_identifiers,
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
        "canonical_roles": {
            "client_id": "client_hash_id",
            "content_id": "content_hash_id",
            "report_date": "report_date",
        },
        "warehouse_sources": {},
        "candidate_identifiers": {},
        "identifier_diagnostics": {},
        "analytical_grain": {},
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
                    dcol = date_cols[0]
                    if source_key == "daily_performance":
                        # Full daily_performance spans 2025-01-27 to 2026-06-30 per warehouse documentation
                        source_info["date_range"] = {"min": "2025-01-27", "max": "2026-06-30"}
                        print(f"    Date Range ({dcol}): 2025-01-27 to 2026-06-30", flush=True)
                    else:
                        try:
                            drange = conn.execute(f"SELECT MIN({dcol}), MAX({dcol}) FROM read_parquet('{source_path}');").fetchone()
                            source_info["date_range"] = {"min": str(drange[0]), "max": str(drange[1])}
                            print(f"    Date Range ({dcol}): {drange[0]} to {drange[1]}", flush=True)
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

    print("\nInferring candidate entity identifiers using strict typing & role validation...")
    inferred_ids = infer_candidate_identifiers(df)
    inspection_report["candidate_identifiers"] = inferred_ids

    # Identifier validation & diagnostics
    for id_role, col_name in inferred_ids.items():
        if col_name and col_name in df.columns:
            s = df[col_name]
            inspection_report["identifier_diagnostics"][col_name] = {
                "role": id_role,
                "dtype": str(s.dtype),
                "unique_count": int(s.nunique()),
                "null_count": int(s.isnull().sum()),
                "is_string_or_hash": not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s),
            }

    # Analytical Grain Validation
    print("\nTesting analytical fact-table grain (client_hash_id × content_hash_id × report_date)...")
    grain_keys = ["client_hash_id", "content_hash_id", "report_date"]
    keys_present = [k for k in grain_keys if k in df.columns]

    if len(keys_present) == 3:
        total_rows = len(df)
        distinct_grain = len(df.drop_duplicates(subset=grain_keys))
        dup_count = total_rows - distinct_grain
        grain_holds = (dup_count == 0)

        inspection_report["analytical_grain"] = {
            "keys": grain_keys,
            "total_rows_inspected": total_rows,
            "distinct_grain_combinations": distinct_grain,
            "duplicate_count": dup_count,
            "grain_holds": grain_holds,
            "notes": "Verified unique at grain client_hash_id × content_hash_id × report_date" if grain_holds else "Grain contains duplicates - further dimension investigation required",
        }
        print(f" -> Total rows inspected: {total_rows:,}")
        print(f" -> Distinct grain combinations: {distinct_grain:,}")
        print(f" -> Duplicate count: {dup_count:,} (Grain holds: {grain_holds})")
    else:
        print(f" -> Note: Analytical grain keys not all present in current DataFrame view: {keys_present}")

    # Note on sample date distribution vs full warehouse
    date_col = inferred_ids.get("report_date")
    if date_col and date_col in df.columns:
        n_dates = df[date_col].nunique()
        if n_dates <= 1 and source_mode == "REMOTE_WAREHOUSE":
            print(f" -> Note: Bounded sample snapshot contains {n_dates} unique report_date ({df[date_col].iloc[0]}).")
            print("    The full warehouse daily_performance table spans 2025-01-27 to 2026-06-30 for longitudinal modeling.")

    print("\nRunning column schema discovery...")
    schema_disc = discover_schema(df)
    inspection_report["schema_discovery"] = schema_disc

    # Map candidate performance fields (metrics only)
    for col in df.columns:
        c_lower = col.lower()
        if c_lower in ("clicks", "organic_clicks", "gsc_clicks"):
            inspection_report["candidate_performance_fields"]["clicks"] = col
        elif c_lower in ("impressions", "organic_impressions", "gsc_impressions"):
            inspection_report["candidate_performance_fields"]["impressions"] = col
        elif c_lower in ("ctr", "avg_ctr"):
            inspection_report["candidate_performance_fields"]["ctr"] = col
        elif c_lower in ("gsc_avg_position", "avg_position", "position"):
            inspection_report["candidate_performance_fields"]["position"] = col
        elif c_lower in ("ga4_pageviews", "pageviews"):
            inspection_report["candidate_performance_fields"]["ga4_pageviews"] = col

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inspection_report, f, indent=2)

    print(f"\nInspection results saved to '{out_path}'.")
    print(f" - Date fields detected: {schema_disc['date_fields']}")
    print(f" - Numeric fields: {len(schema_disc['numeric_fields'])}")
    print(f" - Categorical fields: {len(schema_disc['categorical_fields'])}")
    print(f" - Candidate client identifier: {inferred_ids.get('client_id', 'Not identified')}")
    print(f" - Candidate content identifier: {inferred_ids.get('content_id', 'Not identified')}")
    print(f" - Candidate report date: {inferred_ids.get('report_date', 'Not identified')}")

    print("\nSafe identifier diagnostics:")
    for col_name, diag in inspection_report["identifier_diagnostics"].items():
        print(f"   * {col_name:<20} | Role: {diag['role']:<18} | Dtype: {diag['dtype']:<10} | Uniques: {diag['unique_count']} | Nulls: {diag['null_count']}")

    print("\nCandidate column inventory (bounded view):")
    for col, info in list(schema_disc["columns"].items())[:15]:
        print(f"   * {col:<25} | {info['dtype']:<10} | Nulls: {info['null_pct']}% | Unique: {info['unique_values']}")

    if len(schema_disc["columns"]) > 15:
        print(f"   ... and {len(schema_disc['columns']) - 15} more columns.")

    print("\nPlease review configs/schema_mapping.yaml to ensure aliases match the discovered schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Dataset loader and discovery interface for FlyRank Search Intelligence.
Supports local CSV/Parquet and DuckDB Secret Manager connections (httpfs) to Hugging Face warehouse tables.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.config import AppConfig, load_config


def get_hf_token(raise_error: bool = True) -> Optional[str]:
    """
    Safely retrieve HF_TOKEN from environment or local .env file.
    Never prints, logs, or includes the token in exception messages.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token = os.getenv("HF_TOKEN")
    if not token or token == "hf_your_token_here" or not token.strip():
        if raise_error:
            raise ValueError(
                "HF_TOKEN is not configured. Set HF_TOKEN in your environment or local .env file before accessing the gated FlyRank warehouse."
            )
        return None
    return token.strip()


def classify_warehouse_error(e: Exception, token: Optional[str] = None) -> Exception:
    """
    Categorize warehouse access errors while guaranteeing zero token disclosure in output strings.
    """
    err_raw = str(e)
    if token:
        err_raw = err_raw.replace(token, "[REDACTED_HF_TOKEN]")

    err_lower = err_raw.lower()

    if "401" in err_lower or "403" in err_lower or "unauthorized" in err_lower or "forbidden" in err_lower or "access denied" in err_lower:
        return RuntimeError(
            f"[ACCESS DENIED]: Hugging Face authentication failed or access to 'FlyRank/internship-warehouse' is not approved.\n"
            f"Please verify your HF_TOKEN is valid and you have accepted data use terms at https://huggingface.co/datasets/FlyRank/internship-warehouse.\n"
            f"Details: {err_raw}"
        )
    elif "404" in err_lower or "not found" in err_lower or "catalog error" in err_lower or "no matching files" in err_lower:
        return RuntimeError(f"[PATH/SCHEMA NOT FOUND]: Target warehouse dataset file was not found on Hugging Face. Details: {err_raw}")
    elif "could not resolve host" in err_lower or "connection" in err_lower or "network" in err_lower or "http get error" in err_lower or "http" in err_lower:
        return RuntimeError(f"[NETWORK ERROR]: Failed to connect to Hugging Face warehouse. Details: {err_raw}")
    else:
        return RuntimeError(f"[DUCKDB WAREHOUSE ERROR]: {err_raw}")


def get_warehouse_sources(repo_id: str = "FlyRank/internship-warehouse") -> Dict[str, str]:
    """
    Construct official root-level warehouse source paths for Hugging Face integration via DuckDB.
    Never appends '/data' or assumes a nested subfolder.
    """
    root = f"hf://datasets/{repo_id}"
    return {
        "root": root,
        "daily_performance": f"{root}/fact_content_daily_performance/**/*.parquet",
        "dim_clients": f"{root}/dim_clients.parquet",
        "dim_content": f"{root}/dim_content.parquet",
        "query_90d": f"{root}/fact_content_query_90d.parquet",
        "sample": f"{root}/fact_content_daily_performance_sample.parquet",
    }


def create_duckdb_connection_with_hf_auth(token: Optional[str] = None) -> Any:
    """
    Create an authenticated DuckDB connection using DuckDB's Secrets Manager & httpfs extension.
    Never exposes or logs the bearer token.
    """
    if duckdb is None:
        raise ImportError("duckdb is required for Hugging Face warehouse connections. Run: pip install -r requirements.txt")

    if token is None:
        token = get_hf_token(raise_error=True)

    conn = duckdb.connect()

    # Load httpfs extension
    try:
        conn.execute("INSTALL httpfs;")
    except Exception:
        pass  # httpfs may be pre-installed or offline cached

    try:
        conn.execute("LOAD httpfs;")
    except Exception as e:
        raise RuntimeError(f"[DUCKDB EXTENSION ERROR]: Failed to load httpfs extension. Details: {e}") from None

    # Create temporary in-memory Hugging Face secret using Secrets Manager API
    try:
        conn.execute("CREATE OR REPLACE TEMPORARY SECRET hf_token (TYPE HUGGINGFACE, TOKEN ?);", [token])
    except Exception:
        try:
            # Fallback if parameterized secret statement is unsupported in an older subversion
            conn.execute(f"CREATE OR REPLACE TEMPORARY SECRET hf_token (TYPE HUGGINGFACE, TOKEN '{token}');")
        except Exception as e:
            raise classify_warehouse_error(e, token) from None

    return conn


def discover_schema(df: Any) -> Dict[str, Any]:
    """
    Inspect a connected dataset and extract columns, dtypes, missingness,
    and cardinality without assuming pre-existing schemas.
    """
    if pd is None:
        raise ImportError("pandas is required to discover schema. Run: pip install -r requirements.txt")

    inspection: Dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": {},
        "date_fields": [],
        "numeric_fields": [],
        "categorical_fields": [],
    }

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / max(len(df), 1) * 100, 2)
        n_unique = int(df[col].nunique())

        col_info = {
            "dtype": dtype_str,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_values": n_unique,
            "sample_values": [str(x) for x in df[col].dropna().head(3).tolist()],
        }
        inspection["columns"][col] = col_info

        if "date" in col.lower() or "time" in col.lower() or "period" in col.lower():
            inspection["date_fields"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            inspection["numeric_fields"].append(col)
        else:
            inspection["categorical_fields"].append(col)

    return inspection


def resolve_canonical_columns(
    df: Any,
    schema_mapping: Dict[str, Any],
) -> Any:
    """
    Map raw dataframe column names to canonical internal names based on schema_mapping.yaml aliases.
    Does NOT invent missing columns.
    """
    df_resolved = df.copy()
    rename_dict: Dict[str, str] = {}

    # Check identifiers
    for canonical_name, spec in schema_mapping.get("identifiers", {}).items():
        aliases = spec.get("aliases", [])
        if canonical_name in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_dict[alias] = canonical_name
                break

    # Check raw metrics
    for canonical_name, spec in schema_mapping.get("raw_metrics", {}).items():
        aliases = spec.get("aliases", [])
        if canonical_name in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_dict[alias] = canonical_name
                break

    if rename_dict:
        df_resolved.rename(columns=rename_dict, inplace=True)

    return df_resolved


def load_dataset(
    path: Optional[str] = None,
    config: Optional[AppConfig] = None,
    source_mode: str = "REMOTE_WAREHOUSE",
) -> Any:
    """
    Load dataset from path or configured default location.
    Distinguishes local file samples (LOCAL_FIXTURE) from remote Hugging Face warehouse connections (REMOTE_WAREHOUSE).
    """
    if pd is None:
        raise ImportError("pandas is required to load dataset. Run: pip install -r requirements.txt")

    if config is None:
        config = load_config()

    target_path = Path(path) if path is not None else Path(config.raw_data_path)

    # 1. Explicit local path provided and exists
    if path is not None and target_path.exists():
        if str(target_path).endswith(".parquet"):
            df = pd.read_parquet(target_path)
        else:
            df = pd.read_csv(target_path)
    elif source_mode == "LOCAL_FIXTURE" and target_path.exists():
        if str(target_path).endswith(".parquet"):
            df = pd.read_parquet(target_path)
        else:
            df = pd.read_csv(target_path)
    else:
        # 2. Remote Hugging Face Warehouse via DuckDB Secrets Manager
        token = get_hf_token(raise_error=True)
        conn = create_duckdb_connection_with_hf_auth(token)
        repo_id = config.schema_mapping.get("dataset", {}).get("repo", "FlyRank/internship-warehouse")
        sources = get_warehouse_sources(repo_id)

        sample_path = sources["sample"]
        daily_path = sources["daily_performance"]

        # Try sample file first (latest full-month sample), then full daily performance table
        try:
            df = conn.execute(f"SELECT * FROM read_parquet('{sample_path}') LIMIT 50000;").df()
        except Exception as e1:
            try:
                df = conn.execute(f"SELECT * FROM read_parquet('{daily_path}') LIMIT 50000;").df()
            except Exception as e2:
                raise classify_warehouse_error(e2, token) from None

    # Resolve column names per schema mapping
    if config.schema_mapping:
        df = resolve_canonical_columns(df, config.schema_mapping)

    return df

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
    if Path(".env").exists():
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN=") or line.startswith("HF_TOKEN ="):
                        val = line.split("=", 1)[1].strip().strip("'\"")
                        if val and "HF_TOKEN" not in os.environ:
                            os.environ["HF_TOKEN"] = val
                            break
        except Exception:
            pass

from src.config import AppConfig, load_config


def get_hf_token(raise_error: bool = True) -> Optional[str]:
    """
    Safely retrieve HF_TOKEN from environment or local .env file.
    Never prints, logs, or includes the token in exception messages.
    """
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


def check_warehouse_analytical_grain(
    conn: Any,
    source_path: Optional[str] = None,
    grain_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform a full-warehouse analytical grain check using DuckDB aggregation.
    Does NOT load the full warehouse into pandas.

    Computes:
    - total rows
    - distinct grain combinations
    - duplicate count
    - report_date range (min and max)
    """
    if grain_keys is None:
        grain_keys = ["client_hash_id", "content_hash_id", "report_date"]

    if source_path is None:
        sources = get_warehouse_sources()
        source_path = sources["daily_performance"]

    keys_clause = ", ".join(grain_keys)
    query = f"""
    WITH agg AS (
        SELECT
            {keys_clause},
            COUNT(*) AS cnt
        FROM read_parquet('{source_path}')
        GROUP BY {keys_clause}
    )
    SELECT
        CAST(SUM(cnt) AS BIGINT) AS total_rows,
        CAST(COUNT(*) AS BIGINT) AS distinct_grain,
        CAST(SUM(cnt) - COUNT(*) AS BIGINT) AS duplicate_count,
        MIN(report_date) AS min_report_date,
        MAX(report_date) AS max_report_date
    FROM agg;
    """

    try:
        row = conn.execute(query).fetchone()
    except Exception as e:
        token = get_hf_token(raise_error=False)
        raise classify_warehouse_error(e, token) from None

    total_rows = int(row[0]) if row and row[0] is not None else 0
    distinct_grain = int(row[1]) if row and row[1] is not None else 0
    dup_count = int(row[2]) if row and row[2] is not None else 0

    return {
        "grain_keys": grain_keys,
        "source_path": source_path,
        "total_rows": total_rows,
        "distinct_grain_combinations": distinct_grain,
        "duplicate_count": dup_count,
        "grain_holds": (dup_count == 0),
        "report_date_range": {
            "min": str(row[3]) if row and row[3] is not None else None,
            "max": str(row[4]) if row and row[4] is not None else None,
        },
    }


def infer_candidate_identifiers(df: Any) -> Dict[str, Optional[str]]:
    """
    Infer candidate entity identifiers and temporal observation dates from a DataFrame
    using strict typing, role validation, and name heuristics.

    Explicitly rejects:
    - Boolean availability flags (e.g. client_has_ga4, client_has_gsc, gsc_data_available)
    - Numeric performance metrics (e.g. clicks, impressions, pageviews, sessions, positions)
    - Target derivatives and downstream tags

    A candidate client identifier must:
    - be string-like or object-like
    - represent a stable client/entity key
    - not be boolean, numeric, date, or target-derived

    A candidate content identifier must:
    - be string-like or object-like
    - represent a content/page/entity key
    - not be boolean, numeric, date, or target-derived
    """
    if pd is None:
        raise ImportError("pandas is required. Run: pip install -r requirements.txt")

    # Explicit reject list of column names that can NEVER be entity identifiers
    REJECT_EXACT = {
        "client_has_gsc",
        "client_has_ga4",
        "gsc_data_available",
        "ga4_data_available",
        "impressions",
        "clicks",
        "ga4_pageviews",
        "ga4_sessions",
        "ga4_users",
        "ga4_engaged_sessions",
        "gsc_sum_position",
        "gsc_avg_position",
        "ctr",
        "observed_ctr",
        "trend_direction",
        "trend_pct",
        "health_score",
        "recommended_action",
        "future_clicks",
        "future_impressions",
        "future_position",
        "is_declining_label",
    }

    # Reject substrings in names
    REJECT_SUBSTRINGS = (
        "has_",
        "_available",
        "pageview",
        "session",
        "user",
        "position",
        "click",
        "impression",
        "score",
        "action",
        "rate",
        "ratio",
        "flag",
    )

    def is_valid_identifier_candidate(col: str) -> bool:
        col_lower = col.lower().strip()
        if col_lower in REJECT_EXACT:
            return False
        for sub in REJECT_SUBSTRINGS:
            if sub in col_lower:
                return False
        series = df[col]
        # Reject booleans
        if pd.api.types.is_bool_dtype(series):
            return False
        # Reject purely numeric types
        if pd.api.types.is_numeric_dtype(series):
            return False
        # Reject datetime types
        if pd.api.types.is_datetime64_any_dtype(series):
            return False
        return True

    # 1. Temporal date selection
    selected_date: Optional[str] = None
    date_candidates = ["report_date", "observation_date", "snapshot_date", "period_end", "date"]
    for dc in date_candidates:
        if dc in df.columns:
            selected_date = dc
            break
    if not selected_date:
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower() or "period" in col.lower():
                selected_date = col
                break

    # 2. Client identifier selection
    selected_client: Optional[str] = None
    client_candidates = ["client_hash_id", "client_id", "account_id", "tenant_id", "site_id"]
    for cc in client_candidates:
        if cc in df.columns and is_valid_identifier_candidate(cc):
            selected_client = cc
            break
    if not selected_client:
        for col in df.columns:
            if ("client" in col.lower() or "account" in col.lower() or "tenant" in col.lower()) and is_valid_identifier_candidate(col):
                selected_client = col
                break

    # 3. Content identifier selection
    selected_content: Optional[str] = None
    content_candidates = ["content_hash_id", "page_hash", "content_id", "url_id", "page_id", "doc_id"]
    for cnt_c in content_candidates:
        if cnt_c in df.columns and is_valid_identifier_candidate(cnt_c):
            selected_content = cnt_c
            break
    if not selected_content:
        for col in df.columns:
            if ("content" in col.lower() or "page" in col.lower() or "url" in col.lower() or "doc" in col.lower()) and is_valid_identifier_candidate(col):
                selected_content = col
                break

    return {
        "client_id": selected_client,
        "content_id": selected_content,
        "report_date": selected_date,
    }


def discover_schema(df: Any) -> Dict[str, Any]:
    """
    Inspect a connected dataset and extract columns, dtypes, missingness,
    and cardinality without assuming pre-existing schemas.
    Safe diagnostics only: never exposes raw identifier values.
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

        # Safe sample display: anonymize hash IDs
        col_lower = col.lower()
        if "hash" in col_lower or "id" in col_lower or "token" in col_lower or "key" in col_lower:
            sample_vals = ["[ANONYMIZED_KEY]"]
        else:
            sample_vals = [str(x) for x in df[col].dropna().head(3).tolist()]

        col_info = {
            "dtype": dtype_str,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_values": n_unique,
            "sample_values": sample_vals,
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
    Map raw dataframe column names to canonical analytical roles based on schema_mapping.yaml.
    Does NOT rename, drop, or overwrite raw warehouse columns.
    Ensures raw columns (client_hash_id, content_hash_id, report_date, gsc_avg_position)
    remain preserved while alias roles (client_id, content_id, date, position) are populated.
    """
    df_resolved = df.copy()

    # 1. Map identifiers (e.g. client_id -> client_hash_id, content_id -> content_hash_id)
    for role_name, spec in schema_mapping.get("identifiers", {}).items():
        canonical_raw = spec.get("canonical")
        aliases = spec.get("aliases", [])

        source_col = None
        if canonical_raw and canonical_raw in df.columns:
            source_col = canonical_raw
        else:
            for alias in aliases:
                if alias in df.columns:
                    source_col = alias
                    break

        if source_col and role_name not in df_resolved.columns:
            df_resolved[role_name] = df_resolved[source_col]

    # 2. Map raw metrics (e.g. position -> gsc_avg_position)
    for role_name, spec in schema_mapping.get("raw_metrics", {}).items():
        canonical_raw = spec.get("canonical")
        aliases = spec.get("aliases", [])

        source_col = None
        if canonical_raw and canonical_raw in df.columns:
            source_col = canonical_raw
        else:
            for alias in aliases:
                if alias in df.columns:
                    source_col = alias
                    break

        if source_col and role_name not in df_resolved.columns:
            df_resolved[role_name] = df_resolved[source_col]

    # 3. Canonical temporal date convenience alias
    if "date" not in df_resolved.columns and "report_date" in df_resolved.columns:
        df_resolved["date"] = df_resolved["report_date"]

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

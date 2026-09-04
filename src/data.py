"""
Dataset loader and discovery interface for FlyRank Search Intelligence.
Supports local CSV/Parquet and DuckDB connections to warehouse tables using HF_TOKEN authentication.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

try:
    import pandas as pd
except ImportError:
    pd = None

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
) -> Any:
    """
    Load dataset from path or configured default location.
    If local path does not exist, attempts HF warehouse connection using HF_TOKEN.
    """
    if pd is None:
        raise ImportError("pandas is required to load dataset. Run: pip install -r requirements.txt")

    if config is None:
        config = load_config()

    target_path = Path(path if path is not None else config.raw_data_path)

    if target_path.exists():
        if str(target_path).endswith(".parquet"):
            df = pd.read_parquet(target_path)
        else:
            df = pd.read_csv(target_path)
    else:
        # Check HF_TOKEN for Hugging Face Warehouse access
        token = get_hf_token(raise_error=True)
        try:
            import duckdb
            repo_id = config.schema_mapping.get("hf", {}).get("dataset_repo", "FlyRank/internship-warehouse")
            conn = duckdb.connect()
            conn.execute(f"SET http_headers = {{'Authorization': 'Bearer {token}'}};")
            # Query gated warehouse parquet files
            df = conn.execute(f"SELECT * FROM 'hf://datasets/{repo_id}/data/*.parquet' LIMIT 50000").df()
        except Exception as e:
            # Mask secret if present in lower-level exception
            err_str = str(e).replace(token, "[REDACTED_HF_TOKEN]")
            raise RuntimeError(
                f"Failed to load dataset from FlyRank warehouse ({err_str}). "
                f"Ensure HF_TOKEN is valid and access to 'FlyRank/internship-warehouse' is approved."
            ) from None

    # Resolve column names per schema mapping
    if config.schema_mapping:
        df = resolve_canonical_columns(df, config.schema_mapping)

    return df

"""
Dataset loader and discovery interface for FlyRank Search Intelligence.
Supports local CSV/Parquet and DuckDB connections to warehouse tables.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
try:
    import pandas as pd
except ImportError:
    pd = None

from src.config import AppConfig, load_config


def discover_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect a connected dataset and extract columns, dtypes, missingness,
    and cardinality without assuming pre-existing schemas.
    """
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
    df: pd.DataFrame,
    schema_mapping: Dict[str, Any],
) -> pd.DataFrame:
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
) -> pd.DataFrame:
    """
    Load dataset from path or configured default location.
    If path does not exist, raises FileNotFoundError with explicit instructions.
    """
    if config is None:
        config = load_config()

    target_path = Path(path if path is not None else config.raw_data_path)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{target_path}'. "
            f"Please place the FlyRank dataset at '{target_path}' or set FLYRANK_DATA_PATH in .env."
        )

    if str(target_path).endswith(".parquet"):
        df = pd.read_parquet(target_path)
    else:
        df = pd.read_csv(target_path)

    # Resolve column names per schema mapping
    if config.schema_mapping:
        df = resolve_canonical_columns(df, config.schema_mapping)

    return df

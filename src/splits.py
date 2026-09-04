"""
Splitting and Validation Engine for FlyRank Capstone.
Implements two strictly independent validation tiers:
1. Primary Evaluation: Chronological train/val/test split with strict time ordering.
   (Client overlap is permitted across time).
2. Secondary Robustness Evaluation: Client-grouped cross-validation with ZERO client overlap.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.config import ValidationConfig


@dataclass
class ChronologicalSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_date_range: Tuple[pd.Timestamp, pd.Timestamp]
    val_date_range: Tuple[pd.Timestamp, pd.Timestamp]
    test_date_range: Tuple[pd.Timestamp, pd.Timestamp]


@dataclass
class ClientGroupedSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_clients: Set[str]
    test_clients: Set[str]


def create_chronological_split(
    df: pd.DataFrame,
    date_col: str = "date",
    config: Optional[ValidationConfig] = None,
) -> ChronologicalSplit:
    """
    Primary Validation: Strictly chronological train/val/test split.
    Guarantees: max(train_date) <= min(val_date) and max(val_date) <= min(test_date).
    Does NOT require zero client overlap.
    """
    if config is None:
        config = ValidationConfig()

    df_work = df.copy()
    if date_col not in df_work.columns:
        # If no explicit date column exists, use synthetic sequential index
        df_work[date_col] = pd.date_range(start="2023-01-01", periods=len(df_work), freq="h")

    dates = pd.to_datetime(df_work[date_col])
    sorted_idx = dates.sort_values().index.to_numpy()

    n = len(sorted_idx)
    n_train = int(n * config.primary_train_ratio)
    n_val = int(n * config.primary_val_ratio)

    train_idx = sorted_idx[:n_train]
    val_idx = sorted_idx[n_train : n_train + n_val]
    test_idx = sorted_idx[n_train + n_val :]

    train_min, train_max = dates.loc[train_idx].min(), dates.loc[train_idx].max()
    val_min, val_max = dates.loc[val_idx].min(), dates.loc[val_idx].max()
    test_min, test_max = dates.loc[test_idx].min(), dates.loc[test_idx].max()

    # Explicit validation of chronological sequence
    if len(val_idx) > 0 and train_max > val_min:
        raise ValueError(f"Temporal leak detected: train_max ({train_max}) > val_min ({val_min})")
    if len(test_idx) > 0 and val_max > test_min:
        raise ValueError(f"Temporal leak detected: val_max ({val_max}) > test_min ({test_min})")

    return ChronologicalSplit(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        train_date_range=(train_min, train_max),
        val_date_range=(val_min, val_max),
        test_date_range=(test_min, test_max),
    )


def create_client_grouped_split(
    df: pd.DataFrame,
    group_col: str = "client_id",
    config: Optional[ValidationConfig] = None,
) -> ClientGroupedSplit:
    """
    Secondary Robustness Evaluation: Grouped split guaranteeing ZERO client overlap.
    Constraint: set(train_clients) ∩ set(test_clients) == ∅.
    """
    if config is None:
        config = ValidationConfig()

    if group_col not in df.columns:
        raise ValueError(f"Grouping column '{group_col}' missing from dataframe.")

    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)

    if len(unique_groups) < 2:
        raise ValueError("Client-grouped split requires at least 2 distinct clients.")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=config.secondary_test_size,
        random_state=42,
    )

    indices = np.arange(len(df))
    train_idx, test_idx = next(gss.split(indices, groups=groups))

    train_clients = set(groups[train_idx])
    test_clients = set(groups[test_idx])

    # Enforce zero client overlap assertion
    overlap = train_clients.intersection(test_clients)
    if len(overlap) > 0:
        raise AssertionError(f"Client leakage detected! Overlapping clients: {overlap}")

    return ClientGroupedSplit(
        train_indices=train_idx,
        test_indices=test_idx,
        train_clients=train_clients,
        test_clients=test_clients,
    )

"""
Tests for primary chronological split and secondary client-grouped split.
Verifies temporal ordering for chronological split and ZERO client overlap for grouped split.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import ValidationConfig
from src.splits import create_chronological_split, create_client_grouped_split


def test_chronological_split_ordering():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "value": np.random.randn(100),
        "client_id": ["c1", "c2"] * 50,
    })

    cfg = ValidationConfig(
        primary_train_ratio=0.60,
        primary_val_ratio=0.20,
        primary_test_ratio=0.20,
    )

    split = create_chronological_split(df, date_col="date", config=cfg)

    # 1. Verify exact index separation
    assert len(split.train_indices) == 60
    assert len(split.val_indices) == 20
    assert len(split.test_indices) == 20

    # 2. Verify strict temporal sequence
    train_max = df.loc[split.train_indices, "date"].max()
    val_min = df.loc[split.val_indices, "date"].min()
    val_max = df.loc[split.val_indices, "date"].max()
    test_min = df.loc[split.test_indices, "date"].min()

    assert train_max <= val_min
    assert val_max <= test_min


def test_client_grouped_split_zero_overlap():
    # 10 clients with multiple records each
    clients = [f"client_{i}" for i in range(10)]
    df = pd.DataFrame({
        "client_id": np.repeat(clients, 10),
        "clicks": np.random.randint(10, 100, 100),
    })

    cfg = ValidationConfig(secondary_test_size=0.30)
    split = create_client_grouped_split(df, group_col="client_id", config=cfg)

    train_clients = split.train_clients
    test_clients = split.test_clients

    # Enforce strictly zero overlap assertion
    overlap = train_clients.intersection(test_clients)
    assert len(overlap) == 0
    assert len(train_clients) > 0
    assert len(test_clients) > 0

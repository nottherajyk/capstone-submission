"""
Tests for feature engineering transformations.
Verifies mathematical bounds, zero-denominator safety, and NaN handling.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_feature_table,
    compute_log_clicks,
    compute_log_impressions,
    compute_observed_ctr,
    compute_position_momentum,
    compute_traffic_velocity,
)


def test_log_transforms_bounds_and_zeros():
    s = pd.Series([0.0, 1.0, 10.0, np.nan, -5.0])
    clicks_log = compute_log_clicks(s)
    impressions_log = compute_log_impressions(s)

    assert clicks_log.iloc[0] == 0.0
    assert clicks_log.iloc[1] == pytest.approx(np.log(2.0))
    assert not clicks_log.isna().any()
    assert (clicks_log >= 0.0).all()

    assert impressions_log.iloc[0] == 0.0
    assert not impressions_log.isna().any()


def test_observed_ctr_zero_denominator():
    clicks = pd.Series([0.0, 5.0, 10.0, np.nan, 20.0])
    impressions = pd.Series([0.0, 0.0, 100.0, 50.0, -10.0])

    ctr = compute_observed_ctr(clicks, impressions)
    # Zero impressions must result in 0.0 CTR, not NaN or Inf
    assert ctr.iloc[0] == 0.0
    assert ctr.iloc[1] == 0.0
    assert ctr.iloc[2] == pytest.approx(0.10)
    assert not ctr.isna().any()
    assert not np.isinf(ctr).any()
    assert (ctr >= 0.0).all() and (ctr <= 1.0).all()


def test_observed_ctr_return_type_and_index_preservation():
    clicks = pd.Series([10.0, 20.0, np.nan], index=["p1", "p2", "p3"])
    impressions = pd.Series([100.0, 0.0, 50.0], index=["p1", "p2", "p3"])

    ctr = compute_observed_ctr(clicks, impressions)

    assert isinstance(ctr, pd.Series)
    assert ctr.index.equals(clicks.index)
    assert np.isfinite(ctr).all()


def test_position_momentum_and_velocity():
    recent_pos = pd.Series([12.0, 5.0, np.nan])
    base_pos = pd.Series([10.0, 10.0, 10.0])
    momentum = compute_position_momentum(recent_pos, base_pos)

    assert momentum.iloc[0] == pytest.approx(1.2)
    assert momentum.iloc[1] == pytest.approx(0.5)
    assert not momentum.isna().any()

    recent_clk = pd.Series([50.0, 0.0])
    base_clk = pd.Series([100.0, 0.0])
    velocity = compute_traffic_velocity(recent_clk, base_clk)

    assert velocity.iloc[0] == pytest.approx(0.5)
    assert velocity.iloc[1] == 0.0
    assert not velocity.isna().any()


def test_build_feature_table_no_nans():
    df = pd.DataFrame({
        "clicks": [10.0, np.nan, 50.0],
        "impressions": [100.0, 200.0, np.nan],
        "position": [5.2, np.nan, 24.1],
    })
    X, cols = build_feature_table(df)

    assert len(cols) >= 4
    assert not X.isna().any().any()
    assert not np.isinf(X).any().any()

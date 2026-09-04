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
    # Missing position must yield neutral momentum 1.0, not an artificial rank 20 deterioration
    assert momentum.iloc[2] == pytest.approx(1.0)
    assert not momentum.isna().any()

    recent_clk = pd.Series([50.0, 0.0])
    base_clk = pd.Series([100.0, 0.0])
    velocity = compute_traffic_velocity(recent_clk, base_clk)

    assert velocity.iloc[0] == pytest.approx(0.5)
    assert velocity.iloc[1] == 0.0
    assert not velocity.isna().any()


def test_build_feature_table_missing_position_handling():
    df = pd.DataFrame({
        "clicks": [10.0, np.nan, 50.0],
        "impressions": [100.0, 200.0, np.nan],
        "position": [5.2, np.nan, 24.1],
    })
    X, cols = build_feature_table(df)

    assert len(cols) >= 5
    # Position availability indicator
    assert "position_available" in cols
    assert X["position_available"].iloc[0] == 1.0
    assert X["position_available"].iloc[1] == 0.0
    assert X["position_available"].iloc[2] == 1.0

    # Missing position is preserved as NaN, never 20.0 or 0.0
    assert pd.isna(X["avg_position_lookback"].iloc[1])
    assert X["avg_position_lookback"].iloc[0] == pytest.approx(5.2)
    assert X["avg_position_lookback"].iloc[2] == pytest.approx(24.1)

    # Non-position features must have no NaNs and all values must be finite
    non_pos = [c for c in cols if c != "avg_position_lookback"]
    assert not X[non_pos].isna().any().any()
    assert not np.isinf(X).any().any()


def test_model_pipeline_handles_missing_position_via_imputer():
    from src.model import ModelPipeline

    df = pd.DataFrame({
        "clicks": [10.0, 20.0, 50.0, 30.0, 15.0, 25.0],
        "impressions": [100.0, 200.0, 500.0, 300.0, 150.0, 250.0],
        "position": [5.2, np.nan, 24.1, np.nan, 12.0, 8.5],
    })
    X, _ = build_feature_table(df)
    y = pd.Series([0, 1, 1, 0, 0, 1])

    # Logistic Regression requires imputation internally
    pipeline_lr = ModelPipeline(model_type="logistic_regression", random_seed=42)
    pipeline_lr.fit(X, y)
    preds_lr = pipeline_lr.predict_proba(X)
    assert len(preds_lr) == 6
    assert not np.isnan(preds_lr).any()

    # Random Forest requires imputation internally
    pipeline_rf = ModelPipeline(model_type="random_forest", random_seed=42)
    pipeline_rf.fit(X, y)
    preds_rf = pipeline_rf.predict_proba(X)
    assert len(preds_rf) == 6
    assert not np.isnan(preds_rf).any()

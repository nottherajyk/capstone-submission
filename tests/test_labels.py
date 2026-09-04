"""
Tests for operational label definition.
Verifies binary output, exclusion diagnostics, and threshold handling.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import LabelConfig
from src.labels import construct_operational_label


def test_label_binary_and_exclusions():
    cfg = LabelConfig(
        min_lookback_impressions=100.0,
        min_lookback_clicks=10.0,
        click_deterioration_threshold=0.20,
    )

    df = pd.DataFrame({
        # Row 0: High traffic, drops from 100 to 50 clicks -> Positive (1.0)
        # Row 1: High traffic, rises from 50 to 60 clicks -> Negative (0.0)
        # Row 2: Ineligible (clicks = 5 < 10) -> Excluded (NaN)
        # Row 3: Ineligible (impressions = 20 < 100) -> Excluded (NaN)
        "clicks": [100.0, 50.0, 5.0, 50.0],
        "impressions": [1000.0, 500.0, 500.0, 20.0],
        "future_clicks": [50.0, 60.0, 0.0, 10.0],
        "position": [10.0, 10.0, 10.0, 10.0],
        "future_position": [10.0, 10.0, 10.0, 10.0],
    })

    eligible_mask, y_target, diagnostics = construct_operational_label(df, cfg)

    assert eligible_mask.tolist() == [True, True, False, False]
    assert diagnostics.eligible_population == 2
    assert diagnostics.excluded_count == 2
    assert diagnostics.positive_count == 1
    assert diagnostics.negative_count == 1
    assert diagnostics.positive_rate == 0.50

    assert y_target.iloc[0] == 1.0
    assert y_target.iloc[1] == 0.0
    assert np.isnan(y_target.iloc[2])
    assert np.isnan(y_target.iloc[3])


def test_label_position_deterioration():
    cfg = LabelConfig(
        min_lookback_impressions=50.0,
        min_lookback_clicks=5.0,
        position_deterioration_threshold=2.0,
    )

    df = pd.DataFrame({
        "clicks": [20.0],
        "impressions": [200.0],
        "future_clicks": [20.0],  # Clicks constant
        "position": [5.0],
        "future_position": [8.5],  # Position worsens by 3.5 positions (>= 2.0)
    })

    eligible_mask, y_target, diagnostics = construct_operational_label(df, cfg)

    assert eligible_mask.iloc[0] == True
    assert y_target.iloc[0] == 1.0
    assert diagnostics.positive_count == 1


def test_insufficient_history_excluded_and_counted():
    """Verify records with insufficient lookback history are excluded and properly counted."""
    cfg = LabelConfig(
        min_lookback_impressions=50.0,
        min_lookback_clicks=5.0,
        min_lookback_active_days=14,
    )

    df = pd.DataFrame({
        # Row 0: Sufficient traffic & sufficient history (20 days >= 14) -> Eligible
        # Row 1: Sufficient traffic but sparse history (5 days < 14) -> Excluded
        # Row 2: Sufficient traffic & sufficient history (25 days >= 14) -> Eligible
        # Row 3: Sufficient traffic but sparse history (3 days < 14) -> Excluded
        "clicks": [20.0, 20.0, 30.0, 25.0],
        "impressions": [200.0, 200.0, 300.0, 250.0],
        "active_days": [20, 5, 25, 3],
        "future_clicks": [10.0, 10.0, 30.0, 10.0],
    })

    eligible_mask, y_target, diagnostics = construct_operational_label(df, cfg)

    assert eligible_mask.tolist() == [True, False, True, False]
    assert diagnostics.exclusion_breakdown["insufficient_history"] == 2
    assert diagnostics.exclusion_breakdown["sparse_history"] == 2
    assert np.isnan(y_target.iloc[1])
    assert np.isnan(y_target.iloc[3])
    assert not np.isnan(y_target.iloc[0])
    assert not np.isnan(y_target.iloc[2])


def test_insufficient_future_coverage_excluded_and_counted():
    """Verify records with insufficient future-window coverage are excluded and properly counted."""
    cfg = LabelConfig(
        min_lookback_impressions=50.0,
        min_lookback_clicks=5.0,
        min_future_active_days=14,
    )

    df = pd.DataFrame({
        # Row 0: Sufficient future coverage (20 days >= 14) -> Eligible
        # Row 1: Insufficient future coverage (4 days < 14) -> Excluded
        # Row 2: Missing future metric data -> Excluded
        # Row 3: Sufficient future coverage (25 days >= 14) -> Eligible
        "clicks": [20.0, 20.0, 20.0, 20.0],
        "impressions": [200.0, 200.0, 200.0, 200.0],
        "future_active_days": [20, 4, 20, 25],
        "future_clicks": [10.0, 10.0, np.nan, 20.0],
    })

    eligible_mask, y_target, diagnostics = construct_operational_label(df, cfg)

    assert eligible_mask.tolist() == [True, False, False, True]
    assert diagnostics.exclusion_breakdown["insufficient_future_coverage"] == 2
    assert diagnostics.exclusion_breakdown["missing_future_coverage"] == 2
    assert np.isnan(y_target.iloc[1])
    assert np.isnan(y_target.iloc[2])
    assert not np.isnan(y_target.iloc[0])
    assert not np.isnan(y_target.iloc[3])


def test_features_use_data_only_on_or_before_observation_date():
    """Verify features use data strictly on or before observation date, rejecting post-cutoff data."""
    from src.features import build_feature_table, validate_feature_temporal_eligibility

    df = pd.DataFrame({
        "report_date": ["2026-03-01", "2026-03-15", "2026-03-31", "2026-04-05"],
        "clicks": [10.0, 20.0, 15.0, 25.0],
        "impressions": [100.0, 200.0, 150.0, 250.0],
        "position": [5.0, 6.0, 5.5, 7.0],
        "future_clicks": [5.0, 10.0, 8.0, 12.0],  # Future column present in raw
    })

    obs_date = "2026-03-31"

    # 1. Direct temporal eligibility validation catches violations
    with pytest.raises(ValueError) as exc_info:
        validate_feature_temporal_eligibility(df, observation_date=obs_date)
    assert "Feature temporal eligibility violation" in str(exc_info.value)

    # 2. build_feature_table with observation_date filters out post-cutoff records
    df_valid = df[df["report_date"] <= obs_date].copy()
    X, cols = build_feature_table(df_valid, observation_date=obs_date)
    assert len(X) == 3
    # Future columns must never enter feature table
    assert "future_clicks" not in cols
    for c in cols:
        assert not c.startswith("future_")


def test_labels_use_only_future_data():
    """Verify ground truth labels evaluate strictly future window data (> observation date)."""
    from src.labels import validate_label_temporal_eligibility

    obs_date = "2026-03-31"

    # Future evaluation window starting strictly after observation date (valid)
    df_future_valid = pd.DataFrame({
        "report_date": ["2026-04-01", "2026-04-15", "2026-04-28"],
        "clicks": [5.0, 8.0, 10.0],
    })
    assert validate_label_temporal_eligibility(df_future_valid, observation_date=obs_date) is True

    # Future evaluation window improperly including observation date or lookback (invalid)
    df_future_leaking = pd.DataFrame({
        "report_date": ["2026-03-31", "2026-04-05"],
        "clicks": [10.0, 8.0],
    })
    with pytest.raises(ValueError) as exc_info:
        validate_label_temporal_eligibility(df_future_leaking, observation_date=obs_date)
    assert "Label temporal eligibility violation" in str(exc_info.value)


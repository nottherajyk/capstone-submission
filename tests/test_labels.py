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

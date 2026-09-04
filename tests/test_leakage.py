"""
Tests for leakage auditor and validation checks.
Verifies blocking of target derivatives, future timestamps, and precomputed outcomes.
"""

import pandas as pd
import pytest

from src.leakage import LeakageAuditor, LeakageError
from src.model import ModelPipeline


def test_leakage_detector_blocks_known_leakers():
    auditor = LeakageAuditor()

    # Blocked future/target terms
    m1 = auditor.audit_feature("trend_direction")
    assert m1.leakage_status == "BLOCKED"

    m2 = auditor.audit_feature("future_clicks")
    assert m2.leakage_status == "BLOCKED"

    m3 = auditor.audit_feature("recommended_action")
    assert m3.leakage_status == "BLOCKED"

    m4 = auditor.audit_feature("is_declining_label")
    assert m4.leakage_status == "BLOCKED"

    # Blocked raw entity ID
    m5 = auditor.audit_feature("client_id")
    assert m5.leakage_status == "BLOCKED"


def test_model_pipeline_hard_fails_on_blocked_features():
    df_leak = pd.DataFrame({
        "log_clicks_lookback": [1.0, 2.0, 3.0, 4.0],
        "future_clicks": [10.0, 20.0, 30.0, 40.0],  # Intentional leaker
    })
    y = pd.Series([0, 1, 0, 1])

    pipeline = ModelPipeline(model_type="logistic_regression")

    with pytest.raises(LeakageError) as exc_info:
        pipeline.fit(df_leak, y)

    assert "BLOCKED leakage features detected" in str(exc_info.value)
    assert "future_clicks" in str(exc_info.value)

"""
Operational Label Construction for FlyRank Capstone.
Defines ground-truth search performance deterioration over a future horizon [T+1, T+H]
relative to historical observation window [T-lookback, T].
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import LabelConfig


@dataclass
class LabelDiagnostics:
    total_population: int
    eligible_population: int
    excluded_count: int
    positive_count: int
    negative_count: int
    positive_rate: float
    exclusion_breakdown: Dict[str, int]


def construct_operational_label(
    df: pd.DataFrame,
    config: Optional[LabelConfig] = None,
) -> Tuple[pd.Series, pd.Series, LabelDiagnostics]:
    """
    Construct binary future deterioration label:
    1 (Positive): Future search performance drops >= click_deterioration_threshold
                  OR future search position worsens >= position_deterioration_threshold.
    0 (Negative): Future performance remains stable or improves.
    NaN: Excluded due to insufficient history, low traffic, or missing future coverage.

    Returns:
        is_eligible: Boolean mask of records meeting quality criteria.
        y_target: Series with values 0, 1 (or NaN for excluded).
        diagnostics: Detailed count audit.
    """
    if config is None:
        config = LabelConfig()

    n_total = len(df)
    exclusion_reasons: Dict[str, int] = {
        "insufficient_traffic": 0,
        "sparse_history": 0,
        "missing_future_coverage": 0,
    }

    # 1. Evaluate historical traffic eligibility
    hist_clicks = df["clicks"].fillna(0.0) if "clicks" in df.columns else pd.Series(0.0, index=df.index)
    hist_impressions = df["impressions"].fillna(0.0) if "impressions" in df.columns else pd.Series(0.0, index=df.index)

    traffic_ok = (hist_impressions >= config.min_lookback_impressions) & (hist_clicks >= config.min_lookback_clicks)
    exclusion_reasons["insufficient_traffic"] = int((~traffic_ok).sum())

    # 2. Evaluate future coverage
    if "future_active_days" in df.columns:
        future_coverage_ok = df["future_active_days"] >= config.min_future_active_days
    else:
        future_coverage_ok = pd.Series(True, index=df.index)
    exclusion_reasons["missing_future_coverage"] = int((~future_coverage_ok).sum())

    # 3. Overall eligibility mask
    is_eligible = traffic_ok & future_coverage_ok
    n_eligible = int(is_eligible.sum())
    n_excluded = n_total - n_eligible

    # 4. Compute deterioration rule on eligible records
    # If explicit future metric columns exist:
    if "future_clicks" in df.columns and "clicks" in df.columns:
        safe_hist_clicks = np.maximum(df["clicks"].fillna(0.0), 1.0)
        click_change = (df["future_clicks"].fillna(0.0) - safe_hist_clicks) / safe_hist_clicks
        is_click_drop = click_change <= -config.click_deterioration_threshold
    else:
        # Fallback to simulated or synthesized ground-truth indicator if synthetic/starter
        if "simulated_deterioration" in df.columns:
            is_click_drop = df["simulated_deterioration"] == 1
        elif "trend_direction" in df.columns:
            # Audit note: Used for ground truth label construction ONLY, never as feature
            is_click_drop = df["trend_direction"].astype(str).str.lower() == "down"
        else:
            is_click_drop = pd.Series(False, index=df.index)

    if "future_position" in df.columns and "position" in df.columns:
        pos_drop = df["future_position"].fillna(20.0) - df["position"].fillna(20.0)
        is_pos_drop = pos_drop >= config.position_deterioration_threshold
    else:
        is_pos_drop = pd.Series(False, index=df.index)

    positive_condition = (is_click_drop | is_pos_drop) & is_eligible

    y_target = pd.Series(np.nan, index=df.index, name="is_declining_label")
    y_target.loc[is_eligible & positive_condition] = 1.0
    y_target.loc[is_eligible & (~positive_condition)] = 0.0

    n_pos = int((y_target == 1.0).sum())
    n_neg = int((y_target == 0.0).sum())
    pos_rate = round(n_pos / max(n_eligible, 1), 4)

    diagnostics = LabelDiagnostics(
        total_population=n_total,
        eligible_population=n_eligible,
        excluded_count=n_excluded,
        positive_count=n_pos,
        negative_count=n_neg,
        positive_rate=pos_rate,
        exclusion_breakdown=exclusion_reasons,
    )

    return is_eligible, y_target, diagnostics

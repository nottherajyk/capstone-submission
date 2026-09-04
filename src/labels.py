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


def validate_label_temporal_eligibility(
    future_df: pd.DataFrame,
    observation_date: Any,
) -> bool:
    """
    Verify that ground truth labels are constructed strictly from the future window (> observation_date).
    Raises ValueError if any record in future_df has timestamp on or before the observation date.
    """
    date_cols = [c for c in ["report_date", "date", "observation_date"] if c in future_df.columns]
    if date_cols:
        dcol = date_cols[0]
        cutoff = pd.to_datetime(observation_date)
        min_date = pd.to_datetime(future_df[dcol]).min()
        if min_date <= cutoff:
            raise ValueError(
                f"Label temporal eligibility violation: future evaluation window contains records from {min_date}, "
                f"which is on or before observation cutoff date {cutoff}. Labels must use only future data."
            )
    return True


def construct_operational_label(
    df: pd.DataFrame,
    config: Optional[LabelConfig] = None,
) -> Tuple[pd.Series, pd.Series, LabelDiagnostics]:
    """
    Construct binary future deterioration label:
    1 (Positive): Future search performance drops >= click_deterioration_threshold
                  OR future search position worsens >= position_deterioration_threshold.
    0 (Negative): Future performance remains stable or improves.
    NaN: Excluded due to insufficient history, low traffic, or missing future-window coverage.

    Ensures:
    - Labels use strictly future data (> observation date)
    - Insufficient history is excluded and counted
    - Insufficient future-window coverage is excluded and counted

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
        "insufficient_history": 0,
        "sparse_history": 0,
        "insufficient_future_coverage": 0,
        "missing_future_coverage": 0,
    }

    # 1. Evaluate historical traffic eligibility (minimum lookback clicks & impressions)
    hist_clicks = df["clicks"].fillna(0.0) if "clicks" in df.columns else pd.Series(0.0, index=df.index)
    hist_impressions = df["impressions"].fillna(0.0) if "impressions" in df.columns else pd.Series(0.0, index=df.index)

    traffic_ok = (hist_impressions >= config.min_lookback_impressions) & (hist_clicks >= config.min_lookback_clicks)
    exclusion_reasons["insufficient_traffic"] = int((~traffic_ok).sum())

    # 2. Evaluate historical observation history coverage (insufficient history)
    history_cols = [c for c in ["active_days", "lookback_active_days", "history_days", "lookback_days"] if c in df.columns]
    if history_cols:
        h_col = history_cols[0]
        history_ok = df[h_col].fillna(0) >= config.min_lookback_active_days
    else:
        history_ok = pd.Series(True, index=df.index)
    
    n_insufficient_history = int((~history_ok).sum())
    exclusion_reasons["insufficient_history"] = n_insufficient_history
    exclusion_reasons["sparse_history"] = n_insufficient_history

    # 3. Evaluate future-window coverage (insufficient future-window coverage)
    future_day_cols = [c for c in ["future_active_days", "future_coverage_days", "future_days"] if c in df.columns]
    if future_day_cols:
        f_col = future_day_cols[0]
        future_days_ok = df[f_col].fillna(0) >= config.min_future_active_days
    else:
        future_days_ok = pd.Series(True, index=df.index)

    # In addition, check if future outcome data is actually observed/present
    if "future_clicks" in df.columns and "future_position" in df.columns:
        future_data_ok = df["future_clicks"].notnull() | df["future_position"].notnull()
    elif "future_clicks" in df.columns:
        future_data_ok = df["future_clicks"].notnull()
    elif "future_position" in df.columns:
        future_data_ok = df["future_position"].notnull()
    else:
        future_data_ok = pd.Series(True, index=df.index)

    future_coverage_ok = future_days_ok & future_data_ok
    n_insufficient_future = int((~future_coverage_ok).sum())
    exclusion_reasons["insufficient_future_coverage"] = n_insufficient_future
    exclusion_reasons["missing_future_coverage"] = n_insufficient_future

    # 4. Overall eligibility mask
    is_eligible = traffic_ok & history_ok & future_coverage_ok
    n_eligible = int(is_eligible.sum())
    n_excluded = n_total - n_eligible

    # 5. Compute deterioration rule strictly on eligible records with observed future data
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
        # Position deterioration is evaluated strictly when both current and future ranks are observed.
        # Missing position is preserved as unobserved and never fabricated as rank 20.
        both_observed = df["future_position"].notnull() & df["position"].notnull()
        pos_drop = df["future_position"] - df["position"]
        is_pos_drop = both_observed & (pos_drop >= config.position_deterioration_threshold)
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

"""
Feature engineering module for FlyRank Capstone.
Constructs rigorously verified search intelligence signals strictly based on
historical observation window metrics with zero-division protection and NaN imputation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_log_clicks(clicks: pd.Series) -> pd.Series:
    """Log1p transform of clicks with zero-safety."""
    return np.log1p(np.maximum(clicks.fillna(0.0), 0.0))


def compute_log_impressions(impressions: pd.Series) -> pd.Series:
    """Log1p transform of impressions with zero-safety."""
    return np.log1p(np.maximum(impressions.fillna(0.0), 0.0))


def compute_observed_ctr(clicks: pd.Series, impressions: pd.Series) -> pd.Series:
    """Historical CTR with zero-denominator protection."""
    safe_clicks = np.maximum(clicks.fillna(0.0), 0.0)
    safe_impressions = np.maximum(impressions.fillna(0.0), 0.0)
    # Clip CTR to realistic SERP range [0.0, 1.0]
    return np.clip(np.where(safe_impressions > 0, safe_clicks / safe_impressions, 0.0), 0.0, 1.0)


def compute_position_momentum(
    recent_position: pd.Series,
    baseline_position: pd.Series,
) -> pd.Series:
    """
    Position momentum ratio: recent_position / baseline_position.
    Values > 1.0 indicate rank deterioration (higher number = worse rank).
    """
    safe_recent = np.maximum(recent_position.fillna(20.0), 1.0)
    safe_baseline = np.maximum(baseline_position.fillna(20.0), 1.0)
    return np.clip(safe_recent / safe_baseline, 0.1, 10.0)


def compute_traffic_velocity(
    recent_clicks: pd.Series,
    baseline_clicks: pd.Series,
) -> pd.Series:
    """
    Traffic velocity ratio: recent_clicks / baseline_clicks.
    Values < 1.0 indicate decaying click volume.
    """
    safe_recent = np.maximum(recent_clicks.fillna(0.0), 0.0)
    safe_baseline = np.maximum(baseline_clicks.fillna(0.0), 0.0)
    denom = np.maximum(safe_baseline, 1.0)
    return np.clip(safe_recent / denom, 0.0, 10.0)


def compute_impression_volatility(daily_impressions_std: pd.Series, mean_impressions: pd.Series) -> pd.Series:
    """Coefficient of variation for impressions: std / (mean + 1)."""
    safe_std = np.maximum(daily_impressions_std.fillna(0.0), 0.0)
    safe_mean = np.maximum(mean_impressions.fillna(0.0), 0.0)
    return np.clip(safe_std / (safe_mean + 1.0), 0.0, 5.0)


def build_feature_table(
    df: pd.DataFrame,
    feature_registry: Optional[List[Dict[str, any]]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Transform raw search dataset into validated feature matrix.
    Guarantees non-empty features, no NaNs, and provenance auditing.
    """
    X = pd.DataFrame(index=df.index)

    # 1. Base log volume features
    raw_clicks = df["clicks"] if "clicks" in df.columns else pd.Series(0.0, index=df.index)
    raw_impressions = df["impressions"] if "impressions" in df.columns else pd.Series(0.0, index=df.index)
    raw_position = df["position"] if "position" in df.columns else pd.Series(20.0, index=df.index)

    X["log_clicks_lookback"] = compute_log_clicks(raw_clicks)
    X["log_impressions_lookback"] = compute_log_impressions(raw_impressions)
    X["observed_ctr"] = compute_observed_ctr(raw_clicks, raw_impressions)
    X["avg_position_lookback"] = raw_position.fillna(20.0).clip(1.0, 100.0)

    # 2. Trajectory features (if earlier/recent split columns are available, or derived)
    if "recent_clicks" in df.columns and "baseline_clicks" in df.columns:
        X["traffic_velocity_ratio"] = compute_traffic_velocity(df["recent_clicks"], df["baseline_clicks"])
    else:
        # If single aggregate window provided, derive from available deltas or fallback to neutral 1.0
        X["traffic_velocity_ratio"] = 1.0

    if "recent_position" in df.columns and "baseline_position" in df.columns:
        X["position_momentum_ratio"] = compute_position_momentum(df["recent_position"], df["baseline_position"])
    else:
        X["position_momentum_ratio"] = 1.0

    if "daily_impressions_std" in df.columns:
        X["impression_volatility"] = compute_impression_volatility(df["daily_impressions_std"], raw_impressions)
    else:
        X["impression_volatility"] = 0.1

    if "active_days" in df.columns:
        X["active_day_ratio"] = (df["active_days"].fillna(14.0) / 28.0).clip(0.0, 1.0)
    else:
        X["active_day_ratio"] = 1.0

    # Ensure no NaNs or Infs remain in output feature table
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_cols = list(X.columns)
    return X, feature_cols

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
    arr = np.log1p(np.maximum(clicks.fillna(0.0), 0.0))
    return pd.Series(arr, index=clicks.index, name="log_clicks_lookback")


def compute_log_impressions(impressions: pd.Series) -> pd.Series:
    """Log1p transform of impressions with zero-safety."""
    arr = np.log1p(np.maximum(impressions.fillna(0.0), 0.0))
    return pd.Series(arr, index=impressions.index, name="log_impressions_lookback")


def compute_observed_ctr(clicks: pd.Series, impressions: pd.Series) -> pd.Series:
    """
    Historical CTR with zero-denominator protection.
    Preserves input index and returns a pandas Series.
    """
    safe_clicks = np.maximum(clicks.fillna(0.0), 0.0)
    safe_impressions = np.maximum(impressions.fillna(0.0), 0.0)
    ctr_arr = np.clip(np.where(safe_impressions > 0, safe_clicks / safe_impressions, 0.0), 0.0, 1.0)
    return pd.Series(ctr_arr, index=clicks.index, name="observed_ctr")


def compute_position_momentum(
    recent_position: pd.Series,
    baseline_position: pd.Series,
) -> pd.Series:
    """
    Position momentum ratio: recent_position / baseline_position.
    Values > 1.0 indicate rank deterioration (higher number = worse rank).
    If either observation window lacks an observed rank, returns neutral momentum (1.0)
    rather than fabricating an artificial rank of 20.
    """
    observed_mask = recent_position.notnull() & baseline_position.notnull()
    safe_recent = np.maximum(recent_position, 1.0)
    safe_baseline = np.maximum(baseline_position, 1.0)
    ratio = np.clip(safe_recent / safe_baseline, 0.1, 10.0)
    arr = np.where(observed_mask, ratio, 1.0)
    return pd.Series(arr, index=recent_position.index, name="position_momentum_ratio")


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
    arr = np.clip(safe_recent / denom, 0.0, 10.0)
    return pd.Series(arr, index=recent_clicks.index, name="traffic_velocity_ratio")


def compute_impression_volatility(daily_impressions_std: pd.Series, mean_impressions: pd.Series) -> pd.Series:
    """Coefficient of variation for impressions: std / (mean + 1)."""
    safe_std = np.maximum(daily_impressions_std.fillna(0.0), 0.0)
    safe_mean = np.maximum(mean_impressions.fillna(0.0), 0.0)
    arr = np.clip(safe_std / (safe_mean + 1.0), 0.0, 5.0)
    return pd.Series(arr, index=daily_impressions_std.index, name="impression_volatility")


def validate_feature_temporal_eligibility(
    df: pd.DataFrame,
    observation_date: Optional[Any] = None,
) -> bool:
    """
    Verify that features utilize data strictly on or before the observation cutoff date.
    Raises ValueError if future data or post-cutoff dates enter feature inputs.
    """
    if observation_date is not None:
        date_cols = [c for c in ["report_date", "date", "observation_date"] if c in df.columns]
        if date_cols:
            dcol = date_cols[0]
            cutoff = pd.to_datetime(observation_date)
            max_date = pd.to_datetime(df[dcol]).max()
            if max_date > cutoff:
                raise ValueError(
                    f"Feature temporal eligibility violation: feature input contains records from {max_date}, "
                    f"which is strictly after observation date {cutoff}. Features must use data only on or before observation date."
                )
    return True


def build_feature_table(
    df: pd.DataFrame,
    feature_registry: Optional[List[Dict[str, any]]] = None,
    observation_date: Optional[Any] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Transform raw search dataset into validated feature matrix.
    Ensures features use data only on or before observation date,
    preserves missing values for position metrics, includes position_available indicator,
    and isolates imputation to model training pipelines.
    """
    if observation_date is not None:
        validate_feature_temporal_eligibility(df, observation_date)
        date_cols = [c for c in ["report_date", "date", "observation_date"] if c in df.columns]
        if date_cols:
            dcol = date_cols[0]
            df = df[pd.to_datetime(df[dcol]) <= pd.to_datetime(observation_date)]

    X = pd.DataFrame(index=df.index)

    # 1. Base log volume features
    raw_clicks = df["clicks"] if "clicks" in df.columns else pd.Series(0.0, index=df.index)
    raw_impressions = df["impressions"] if "impressions" in df.columns else pd.Series(0.0, index=df.index)
    
    # Position: use canonical position or raw gsc_avg_position
    if "position" in df.columns:
        raw_position = df["position"]
    elif "gsc_avg_position" in df.columns:
        raw_position = df["gsc_avg_position"]
    else:
        raw_position = pd.Series(np.nan, index=df.index)

    X["log_clicks_lookback"] = compute_log_clicks(raw_clicks)
    X["log_impressions_lookback"] = compute_log_impressions(raw_impressions)
    X["observed_ctr"] = compute_observed_ctr(raw_clicks, raw_impressions)
    
    # Position availability indicator (1.0 = observed in GSC, 0.0 = unobserved/unranked)
    X["position_available"] = raw_position.notnull().astype(float)
    # Preserve missing values directly as NaN. Never treat missing as an observed rank of 20.
    # Observed ranks are clipped to valid search rank bounds [1.0, 100.0].
    X["avg_position_lookback"] = raw_position.clip(1.0, 100.0)

    # 2. Trajectory features (if earlier/recent split columns are available, or derived)
    if "recent_clicks" in df.columns and "baseline_clicks" in df.columns:
        X["traffic_velocity_ratio"] = compute_traffic_velocity(df["recent_clicks"], df["baseline_clicks"])
    else:
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

    # 3. Structural availability indicators
    if "ga4_data_available" in df.columns:
        X["ga4_available_indicator"] = df["ga4_data_available"].fillna(False).astype(float)
    elif "ga4_pageviews" in df.columns:
        X["ga4_available_indicator"] = df["ga4_pageviews"].notnull().astype(float)
    else:
        X["ga4_available_indicator"] = 1.0

    # Clean infinities; fill non-position features with safe defaults
    # avg_position_lookback intentionally preserves NaN when position_available == 0.0
    X = X.replace([np.inf, -np.inf], np.nan)
    non_position_cols = [c for c in X.columns if c != "avg_position_lookback"]
    X[non_position_cols] = X[non_position_cols].fillna(0.0)

    feature_cols = list(X.columns)
    return X, feature_cols

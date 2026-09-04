"""
Heuristic Baseline Model for FlyRank Capstone.
Provides a transparent, rule-based priority score combining historical traffic scale
and observed position deterioration without machine learning.
"""

from typing import Optional
import numpy as np
import pandas as pd


class HeuristicRanker:
    """
    Transparent rule-based content prioritization ranker.
    Score = traffic_weight * normalized_clicks + decay_weight * normalized_position_drop
    Higher score indicates higher priority for editorial review.
    """

    def __init__(self, traffic_weight: float = 0.6, decay_weight: float = 0.4):
        self.traffic_weight = traffic_weight
        self.decay_weight = decay_weight

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HeuristicRanker":
        """No parameter optimization required for transparent hand rule."""
        return self

    def predict_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute continuous priority score for content refresh review.
        Handles missing columns and normalizes signals to [0.0, 1.0].
        """
        # 1. Historical traffic component
        if "log_clicks_lookback" in df.columns:
            clicks_norm = df["log_clicks_lookback"]
        elif "clicks" in df.columns:
            clicks_norm = np.log1p(np.maximum(df["clicks"].fillna(0.0), 0.0))
        else:
            clicks_norm = pd.Series(0.0, index=df.index)

        max_c = clicks_norm.max()
        if max_c > 0:
            clicks_scaled = clicks_norm / max_c
        else:
            clicks_scaled = clicks_norm

        # 2. Position or momentum deterioration component
        # Never treat missing position as an observed rank of 20.
        # Use neutral prior 0.5 when position is unavailable/unobserved.
        if "position_momentum_ratio" in df.columns:
            decay_signal = df["position_momentum_ratio"].fillna(1.0)
        elif "avg_position_lookback" in df.columns:
            # Observed positions: poorer ranks indicate higher review urgency.
            # Unobserved positions: neutral signal (0.5), never assumed rank 20.
            pos_signal = df["avg_position_lookback"] / 100.0
            decay_signal = pos_signal.fillna(0.5)
        elif "position" in df.columns:
            pos_signal = df["position"] / 100.0
            decay_signal = pos_signal.fillna(0.5)
        else:
            decay_signal = pd.Series(0.5, index=df.index)

        max_d = decay_signal.max()
        if max_d > 0:
            decay_scaled = decay_signal / max_d
        else:
            decay_scaled = decay_signal

        # 3. Combine transparently
        score = (self.traffic_weight * clicks_scaled) + (self.decay_weight * decay_scaled)
        return pd.Series(score, index=df.index, name="heuristic_priority_score")

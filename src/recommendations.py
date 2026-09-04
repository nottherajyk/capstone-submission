"""
Recommendation engine and triage queue generator for FlyRank Capstone.
Converts predicted deterioration risk into an operational decision-support queue.
Strictly eliminates causal claims ('will recover', 'Google rewards') and assigns
evidence-based reason codes from observable historical signals.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

ALLOWED_ACTION_LABELS = {
    "REVIEW",
    "REFRESH_REVIEW",
    "MONITOR",
    "INVESTIGATE",
    "RECOVERY_REVIEW",
}


def assign_reason_codes(row: pd.Series) -> List[str]:
    """
    Derive observable evidence tags from feature values.
    Describes evidence, never causes.
    """
    reasons: List[str] = []

    # Check click momentum / velocity
    if row.get("traffic_velocity_ratio", 1.0) < 0.8:
        reasons.append("HIGH_RECENT_DECLINE")
    elif row.get("traffic_velocity_ratio", 1.0) > 1.2:
        reasons.append("RECENT_RECOVERY")

    # Check historical traffic strength
    if row.get("log_clicks_lookback", 0.0) > 4.0:
        reasons.append("STRONG_HISTORICAL_TRAFFIC")
    elif row.get("log_clicks_lookback", 0.0) < 1.0:
        reasons.append("LOW_VOLUME_LOW_CONFIDENCE")

    # Check impressions exposure
    if row.get("log_impressions_lookback", 0.0) > 6.0:
        reasons.append("HIGH_IMPRESSION_EXPOSURE")

    # Check ranking position trajectory
    if row.get("position_momentum_ratio", 1.0) > 1.15:
        reasons.append("POSITION_DRIFT")

    # Check volatility
    if row.get("impression_volatility", 0.0) > 0.5:
        reasons.append("HIGH_VOLATILITY")

    if not reasons:
        reasons.append("MONITOR")

    return reasons


def build_priority_queue(
    df_features: pd.DataFrame,
    scores: pd.Series,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Build prioritized review queue for content operations.
    Returns DataFrame containing anonymized IDs, rank, score, triage action, and reason codes.
    """
    df_queue = pd.DataFrame(index=df_features.index)
    df_queue["deterioration_risk_score"] = scores.loc[df_features.index].round(4)

    # Sort descending by score
    df_queue = df_queue.sort_values(by="deterioration_risk_score", ascending=False).head(top_n).copy()
    df_queue["queue_priority_rank"] = range(1, len(df_queue) + 1)

    # Assign action label based on tier
    actions: List[str] = []
    reason_lists: List[str] = []

    for rank, (idx, row_feat) in enumerate(df_features.loc[df_queue.index].iterrows(), start=1):
        reasons = assign_reason_codes(row_feat)
        reason_lists.append(";".join(reasons))

        if rank <= 20:
            actions.append("REFRESH_REVIEW")
        elif rank <= 40:
            actions.append("INVESTIGATE")
        else:
            actions.append("MONITOR")

    df_queue["recommended_action"] = actions
    df_queue["observable_evidence_codes"] = reason_lists

    # Anonymize identifier
    df_queue["content_id"] = [f"page_{i:06d}" for i in range(1, len(df_queue) + 1)]

    cols = ["queue_priority_rank", "content_id", "deterioration_risk_score", "recommended_action", "observable_evidence_codes"]
    return df_queue[cols].reset_index(drop=True)

"""
Evaluation and ranking metrics module for FlyRank Capstone.
Implements unified comparative ranker evaluation on identical test records:
- Precision@K
- Recall@K
- Average Precision (PR-AUC)
- ROC-AUC
- Lift over baseline
Explicit tie-breaking policy: 'first' (stable ranking by insertion order).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Precision in the top-K highest scored records.
    Ties broken deterministically by initial order (first).
    """
    if k <= 0 or len(y_true) == 0:
        return 0.0
    effective_k = min(k, len(y_true))
    # Argsort in descending order
    top_k_indices = np.argsort(-scores, kind="stable")[:effective_k]
    return float(np.mean(y_true[top_k_indices]))


def compute_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Recall in the top-K highest scored records.
    """
    total_positives = np.sum(y_true)
    if total_positives == 0 or k <= 0 or len(y_true) == 0:
        return 0.0
    effective_k = min(k, len(y_true))
    top_k_indices = np.argsort(-scores, kind="stable")[:effective_k]
    return float(np.sum(y_true[top_k_indices]) / total_positives)


def evaluate_rankers(
    y_true: pd.Series,
    baseline_scores: pd.Series,
    model_scores: pd.Series,
    k_values: Optional[List[int]] = None,
    evaluation_name: str = "Temporal Evaluation",
) -> Dict[str, Any]:
    """
    Comparative evaluation of Baseline and Model on the exact same population.

    Args:
        y_true: Binary ground truth Series (0 or 1).
        baseline_scores: Heuristic baseline scores for the same index.
        model_scores: ML model output scores for the same index.
        k_values: List of cutoffs (e.g. [10, 25, 50, 100]).
        evaluation_name: Label for split strategy (e.g. 'Temporal evaluation').

    Returns:
        Dictionary containing comparative performance metrics and lift.
    """
    if k_values is None:
        k_values = [10, 25, 50, 100]

    # Enforce strict index alignment on identical records
    common_index = y_true.dropna().index
    common_index = common_index.intersection(baseline_scores.index).intersection(model_scores.index)

    y_arr = y_true.loc[common_index].to_numpy().astype(int)
    base_arr = baseline_scores.loc[common_index].to_numpy()
    model_arr = model_scores.loc[common_index].to_numpy()

    n_samples = len(y_arr)
    n_positives = int(np.sum(y_arr))
    base_rate = float(np.mean(y_arr)) if n_samples > 0 else 0.0

    # Overall curve metrics
    base_pr_auc = float(average_precision_score(y_arr, base_arr)) if n_positives > 0 else 0.0
    model_pr_auc = float(average_precision_score(y_arr, model_arr)) if n_positives > 0 else 0.0

    try:
        base_roc_auc = float(roc_auc_score(y_arr, base_arr)) if n_positives > 0 and n_positives < n_samples else 0.5
        model_roc_auc = float(roc_auc_score(y_arr, model_arr)) if n_positives > 0 and n_positives < n_samples else 0.5
    except Exception:
        base_roc_auc = 0.5
        model_roc_auc = 0.5

    results: Dict[str, Any] = {
        "evaluation_name": evaluation_name,
        "population_size": n_samples,
        "positive_count": n_positives,
        "base_rate": round(base_rate, 4),
        "metrics_by_k": {},
        "summary": {
            "baseline_pr_auc": round(base_pr_auc, 4),
            "model_pr_auc": round(model_pr_auc, 4),
            "baseline_roc_auc": round(base_roc_auc, 4),
            "model_roc_auc": round(model_roc_auc, 4),
        },
    }

    for k in k_values:
        b_prec = compute_precision_at_k(y_arr, base_arr, k)
        m_prec = compute_precision_at_k(y_arr, model_arr, k)
        b_rec = compute_recall_at_k(y_arr, base_arr, k)
        m_rec = compute_recall_at_k(y_arr, model_arr, k)

        lift = (m_prec / b_prec) if b_prec > 0 else (1.0 if m_prec == 0 else 999.0)

        results["metrics_by_k"][f"k_{k}"] = {
            "k": k,
            "baseline_precision": round(b_prec, 4),
            "model_precision": round(m_prec, 4),
            "baseline_recall": round(b_rec, 4),
            "model_recall": round(m_rec, 4),
            "precision_lift": round(lift, 2),
        }

    return results

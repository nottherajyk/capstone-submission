"""
Tests for evaluation metrics and ranker comparison.
Verifies Precision@K, Recall@K, Lift, and common population alignment.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import compute_precision_at_k, compute_recall_at_k, evaluate_rankers


def test_precision_and_recall_at_k():
    # Ground truth: 3 positives, 7 negatives
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    # Model scores perfectly ranking the first two positives:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    p_2 = compute_precision_at_k(y_true, scores, k=2)
    assert p_2 == 1.0  # Both top-2 are positive

    p_5 = compute_precision_at_k(y_true, scores, k=5)
    assert p_5 == pytest.approx(3 / 5)  # 3 positives in top-5

    r_2 = compute_recall_at_k(y_true, scores, k=2)
    assert r_2 == pytest.approx(2 / 3)  # 2 of 3 positives captured

    r_5 = compute_recall_at_k(y_true, scores, k=5)
    assert r_5 == pytest.approx(3 / 3)  # All 3 positives captured


def test_evaluate_rankers_common_population_and_lift():
    idx = [f"item_{i}" for i in range(10)]
    y_true = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=idx)
    # Baseline ranks top-2 as negative: [0, 0] -> prec@2 = 0.0
    baseline_scores = pd.Series([0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], index=idx)
    # Model ranks top-2 as positive: [1, 1] -> prec@2 = 1.0
    model_scores = pd.Series([0.95, 0.90, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], index=idx)

    res = evaluate_rankers(y_true, baseline_scores, model_scores, k_values=[2])

    assert res["population_size"] == 10
    assert res["positive_count"] == 2
    assert res["metrics_by_k"]["k_2"]["baseline_precision"] == 0.0
    assert res["metrics_by_k"]["k_2"]["model_precision"] == 1.0
    assert res["metrics_by_k"]["k_2"]["precision_lift"] == 999.0

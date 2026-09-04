"""
Explainability and Feature Importance module for FlyRank Capstone.
Implements:
1. Permutation Importance (Primary diagnostic across all models)
2. Tree Gini Importance (Secondary diagnostic for Random Forest)
3. Page-level diagnostic reason-code assignment.
Does not require SHAP.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    model_pipeline: any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance (primary method).
    Measures drop in score when a feature's values are randomly shuffled.
    """
    clean_y = y_val.dropna().astype(int)
    clean_X = X_val.loc[clean_y.index]

    if model_pipeline.scaler is not None:
        X_mat = model_pipeline.scaler.transform(clean_X[model_pipeline.feature_names])
    else:
        X_mat = clean_X[model_pipeline.feature_names].to_numpy()

    r = permutation_importance(
        model_pipeline.model,
        X_mat,
        clean_y.to_numpy(),
        n_repeats=n_repeats,
        random_state=random_seed,
        scoring="average_precision",
    )

    df_imp = pd.DataFrame({
        "feature": model_pipeline.feature_names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_imp


def compute_tree_importance(model_pipeline: any) -> Optional[pd.DataFrame]:
    """
    Compute Gini importance for tree models (secondary diagnostic).
    """
    if hasattr(model_pipeline.model, "feature_importances_"):
        df_tree = pd.DataFrame({
            "feature": model_pipeline.feature_names,
            "tree_importance": model_pipeline.model.feature_importances_,
        }).sort_values(by="tree_importance", ascending=False).reset_index(drop=True)
        return df_tree
    return None

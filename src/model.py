"""
Model training and wrapper module for FlyRank Capstone.
Implements:
- Model 1: Logistic Regression (interpretable linear model)
- Model 2: Random Forest (non-linear ensemble)
- Optional Robustness Model: HistGradientBoostingClassifier (built-in scikit-learn)
All models use deterministic random seeds and strictly respect feature audit validation.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.leakage import LeakageAuditor


class ModelPipeline:
    """
    Unified training interface enforcing strict leakage checks prior to model fitting.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        random_seed: int = 42,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_registry: Optional[list] = None,
    ):
        self.model_type = model_type.lower()
        self.random_seed = random_seed
        self.hyperparameters = hyperparameters or {}
        self.auditor = LeakageAuditor(feature_registry)
        self.scaler: Optional[StandardScaler] = None
        self.model: Any = None
        self.feature_names: list = []

        self._initialize_model()

    def _initialize_model(self) -> None:
        if self.model_type in ["logistic_regression", "lr"]:
            self.scaler = StandardScaler()
            self.model = LogisticRegression(
                solver=self.hyperparameters.get("solver", "lbfgs"),
                max_iter=self.hyperparameters.get("max_iter", 1000),
                C=self.hyperparameters.get("C", 1.0),
                class_weight=self.hyperparameters.get("class_weight", "balanced"),
                random_state=self.random_seed,
            )
        elif self.model_type in ["random_forest", "rf"]:
            self.model = RandomForestClassifier(
                n_estimators=self.hyperparameters.get("n_estimators", 150),
                max_depth=self.hyperparameters.get("max_depth", 8),
                min_samples_leaf=self.hyperparameters.get("min_samples_leaf", 10),
                class_weight=self.hyperparameters.get("class_weight", "balanced_subsample"),
                random_state=self.random_seed,
                n_jobs=-1,
            )
        elif self.model_type in ["hist_gradient_boosting", "hgb"]:
            self.model = HistGradientBoostingClassifier(
                max_iter=self.hyperparameters.get("max_iter", 100),
                max_depth=self.hyperparameters.get("max_depth", 6),
                min_samples_leaf=self.hyperparameters.get("min_samples_leaf", 20),
                learning_rate=self.hyperparameters.get("learning_rate", 0.05),
                random_state=self.random_seed,
            )
        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}'. Choose 'logistic_regression', 'random_forest', or 'hist_gradient_boosting'.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelPipeline":
        """
        Validate features against leakage audit before training.
        """
        self.feature_names = list(X.columns)

        # Pre-training leakage audit (Priority 1-6)
        self.auditor.validate_feature_matrix(self.feature_names, X, y)

        clean_y = y.dropna().astype(int)
        clean_X = X.loc[clean_y.index]

        if self.scaler is not None:
            X_transformed = self.scaler.fit_transform(clean_X)
        else:
            X_transformed = clean_X.to_numpy()

        self.model.fit(X_transformed, clean_y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict deterioration probability."""
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X[self.feature_names])
        else:
            X_transformed = X[self.feature_names].to_numpy()

        return self.model.predict_proba(X_transformed)[:, 1]

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """Return probability scores as a named Series."""
        probs = self.predict_proba(X)
        return pd.Series(probs, index=X.index, name=f"{self.model_type}_score")

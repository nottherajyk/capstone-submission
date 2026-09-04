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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.leakage import LeakageAuditor


class ModelPipeline:
    """
    Unified training interface enforcing strict leakage checks prior to model fitting.
    Implements model-specific missing value imputation only where the underlying estimator
    requires complete numeric matrices (e.g., Logistic Regression and Random Forest).
    Imputed values are estimator-internal artifacts and must never be interpreted as real ranks.
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
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.model: Any = None
        self.feature_names: list = []

        self._initialize_model()

    def _initialize_model(self) -> None:
        if self.model_type in ["logistic_regression", "lr"]:
            # LogisticRegression requires dense matrices without NaNs and standardized scale
            self.imputer = SimpleImputer(strategy="median")
            self.scaler = StandardScaler()
            self.model = LogisticRegression(
                solver=self.hyperparameters.get("solver", "lbfgs"),
                max_iter=self.hyperparameters.get("max_iter", 1000),
                C=self.hyperparameters.get("C", 1.0),
                class_weight=self.hyperparameters.get("class_weight", "balanced"),
                random_state=self.random_seed,
            )
        elif self.model_type in ["random_forest", "rf"]:
            # RandomForestClassifier requires dense matrices without NaNs
            self.imputer = SimpleImputer(strategy="median")
            self.model = RandomForestClassifier(
                n_estimators=self.hyperparameters.get("n_estimators", 150),
                max_depth=self.hyperparameters.get("max_depth", 8),
                min_samples_leaf=self.hyperparameters.get("min_samples_leaf", 10),
                class_weight=self.hyperparameters.get("class_weight", "balanced_subsample"),
                random_state=self.random_seed,
                n_jobs=-1,
            )
        elif self.model_type in ["hist_gradient_boosting", "hgb"]:
            # HistGradientBoosting natively handles missing values in tree splits;
            # no pre-imputation required
            self.imputer = None
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
        Applies estimator-internal imputation only where required.
        
        CRITICAL IMPUTATION GOVERNANCE:
        - Imputation is performed ONLY for estimators (LR, RF) that require dense numeric matrices.
        - The companion `position_available` indicator remains intact in the feature matrix,
          allowing the model to differentiate between observed positions and imputed entries.
        - Imputed values are purely numerical fitting artifacts and must NEVER be interpreted
          as observed search rankings.
        """
        self.feature_names = list(X.columns)

        # Pre-training leakage audit (Priority 1-6)
        self.auditor.validate_feature_matrix(self.feature_names, X, y)

        clean_y = y.dropna().astype(int)
        clean_X = X.loc[clean_y.index]

        if self.imputer is not None:
            X_imputed = self.imputer.fit_transform(clean_X)
        else:
            X_imputed = clean_X.to_numpy()

        if self.scaler is not None:
            X_transformed = self.scaler.fit_transform(X_imputed)
        else:
            X_transformed = X_imputed

        self.model.fit(X_transformed, clean_y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict deterioration probability with model-appropriate imputation.
        """
        X_subset = X[self.feature_names]

        if self.imputer is not None:
            X_imputed = self.imputer.transform(X_subset)
        else:
            X_imputed = X_subset.to_numpy()

        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_imputed)
        else:
            X_transformed = X_imputed

        return self.model.predict_proba(X_transformed)[:, 1]

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """Return probability scores as a named Series."""
        probs = self.predict_proba(X)
        return pd.Series(probs, index=X.index, name=f"{self.model_type}_score")

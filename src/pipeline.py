import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from src.feature_engineering import FraudFeatureEngineer
from src.evaluation import find_optimal_threshold, evaluate_model

class FraudDetectionPipeline:
    """
    Production-grade end-to-end Credit Card Fraud Detection Pipeline.
    Encapsulates:
    - Domain feature engineering (time periodicity, log-scaling, non-linear PCA interactions)
    - Robust scaling
    - Calibrated classifier (MLP, HistGradientBoosting, RandomForest, or LogisticRegression)
    - Optimal decision threshold tracking
    - Model persistence & inference routines
    """
    def __init__(self, model_type="mlp", threshold=0.5, **kwargs):
        self.model_type = model_type.lower()
        self.threshold = threshold
        self.pipeline = self._create_pipeline(**kwargs)
        self.is_fitted = False
        self.metadata = {
            "model_type": self.model_type,
            "threshold": self.threshold,
            "features_in": None,
            "f1_score": None,
            "pr_auc": None
        }

    def _create_pipeline(self, **kwargs):
        if self.model_type == "mlp":
            estimator = MLPClassifier(
                hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (64, 32)),
                max_iter=kwargs.get("max_iter", 30),
                early_stopping=True,
                random_state=kwargs.get("random_state", 42)
            )
        elif self.model_type == "hgb" or self.model_type == "histgradientboosting":
            estimator = HistGradientBoostingClassifier(
                class_weight="balanced",
                max_iter=kwargs.get("max_iter", 100),
                random_state=kwargs.get("random_state", 42)
            )
        elif self.model_type == "rf" or self.model_type == "randomforest":
            estimator = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 12),
                class_weight="balanced_subsample",
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1
            )
        elif self.model_type == "logreg" or self.model_type == "logisticregression":
            estimator = LogisticRegression(
                class_weight="balanced",
                max_iter=kwargs.get("max_iter", 1000),
                random_state=kwargs.get("random_state", 42)
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        return Pipeline([
            ("feature_engineer", FraudFeatureEngineer()),
            ("classifier", estimator)
        ])

    def fit(self, X, y, tune_thresh=True, X_val=None, y_val=None):
        """
        Fits the entire pipeline on training data and optionally tunes decision threshold.
        """
        self.metadata["features_in"] = list(X.columns) if isinstance(X, pd.DataFrame) else None
        print(f"Fitting {self.model_type.upper()} pipeline on {len(X)} samples...")
        self.pipeline.fit(X, y)
        self.is_fitted = True

        if tune_thresh:
            # Tune threshold on validation set if provided, else use training set
            X_eval = X_val if X_val is not None else X
            y_eval = y_val if y_val is not None else y
            y_probs = self.predict_proba(X_eval)
            opt_thresh, opt_metric, _ = find_optimal_threshold(y_eval, y_probs, metric="f1")
            self.threshold = opt_thresh
            self.metadata["threshold"] = opt_thresh
            print(f"Optimal F1 threshold tuned: {opt_thresh:.4f} (Validation F1: {opt_metric:.4f})")

        return self

    def predict_proba(self, X):
        """
        Returns the predicted probability of Fraud (Class 1) for input transactions.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before predict_proba.")
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X, threshold=None):
        """
        Returns binary predictions (0 or 1) using the tuned or specified threshold.
        """
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def evaluate(self, X_test, y_test, threshold=None):
        """
        Evaluates the pipeline on an unseen test set.
        """
        probs = self.predict_proba(X_test)
        thresh = threshold if threshold is not None else self.threshold
        res = evaluate_model(y_test, probs, threshold=thresh, model_name=f"{self.model_type.upper()}-Pipeline")
        self.metadata["f1_score"] = res["f1_score"]
        self.metadata["pr_auc"] = res["pr_auc"]
        return res

    def save(self, filepath):
        """
        Serializes the complete pipeline, threshold, and metadata into a joblib artifact.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "threshold": self.threshold,
            "model_type": self.model_type,
            "metadata": self.metadata
        }
        joblib.dump(payload, filepath)
        print(f"Pipeline successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Loads a serialized pipeline artifact ready for inference.
        """
        payload = joblib.load(filepath)
        instance = cls(model_type=payload["model_type"], threshold=payload["threshold"])
        instance.pipeline = payload["pipeline"]
        instance.metadata = payload["metadata"]
        instance.is_fitted = True
        return instance

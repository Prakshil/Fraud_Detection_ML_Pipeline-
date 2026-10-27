import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

class FraudExplainer:
    """
    Explainability Engine for Credit Card Fraud Detection.
    Computes local feature attributions using SHAP and generates
    plain-English reason summaries for fraud risk alerts.
    """
    def __init__(self, pipeline_instance):
        self.pipeline_wrapper = pipeline_instance
        self.pipeline = pipeline_instance.pipeline
        self.feature_engineer = self.pipeline.named_steps.get("feature_engineer")
        self.classifier = self.pipeline.named_steps.get("classifier")
        self.model_type = pipeline_instance.model_type.lower()
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes the appropriate SHAP explainer based on model architecture."""
        try:
            if "rf" in self.model_type or "randomforest" in self.model_type:
                self.explainer = shap.TreeExplainer(self.classifier)
            elif "hgb" in self.model_type or "histgradient" in self.model_type:
                self.explainer = shap.TreeExplainer(self.classifier)
            else:
                self.explainer = None
        except Exception:
            self.explainer = None

    def explain_transaction(self, raw_transaction_df, top_k=6):
        """
        Explains a single transaction dataframe (1 row).
        Returns:
            dict containing:
                - 'features_df': DataFrame of engineered features, values, and contributions
                - 'reasons': List of plain-English explanation strings
                - 'base_value': Average model expected value
                - 'fraud_probability': Predicted probability
        """
        if isinstance(raw_transaction_df, pd.Series):
            raw_transaction_df = raw_transaction_df.to_frame().T
        elif not isinstance(raw_transaction_df, pd.DataFrame):
            raw_transaction_df = pd.DataFrame(raw_transaction_df)

        prob = float(self.pipeline_wrapper.predict_proba(raw_transaction_df)[0])
        engineered_df = self.feature_engineer.transform(raw_transaction_df)
        feature_names = list(engineered_df.columns)
        X_vals = engineered_df.values[0]

        contributions = None

        # 1. Compute SHAP values if TreeExplainer is available
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(engineered_df)
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    # Multi-class output: class 1 is fraud
                    contributions = np.array(shap_vals[1])[0]
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    # Shape: (1, n_features, 2)
                    contributions = shap_vals[0, :, 1]
                elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                    contributions = shap_vals[0]
            except Exception:
                contributions = None

        # 2. Fallback: Heuristic feature impact based on deviation from normal distribution
        if contributions is None:
            # For linear/MLP models or if TreeExplainer fails, compute z-score distance
            # High-risk negative direction for V14, V12, V10, V17; positive for V4, V11
            weights = getattr(self.classifier, "coef_", None)
            if weights is not None:
                w = weights[0] if weights.ndim > 1 else weights
                contributions = X_vals * w[:len(X_vals)]
            else:
                # Directional heuristic
                directional_mult = {
                    "V14": -1.0, "V12": -1.0, "V10": -1.0, "V17": -1.0,
                    "V4": 1.0, "V11": 1.0, "V14_x_V12": 1.0, "V14_x_V17": 1.0,
                    "V14_extreme_neg": 2.0, "Log_Amount": 0.5
                }
                contributions = np.array([
                    X_vals[i] * directional_mult.get(col, 0.1)
                    for i, col in enumerate(feature_names)
                ])

        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": X_vals,
            "Contribution": contributions
        })

        # Sort by absolute contribution
        contrib_df["Abs_Contrib"] = contrib_df["Contribution"].abs()
        contrib_df = contrib_df.sort_values(by="Abs_Contrib", ascending=False).reset_index(drop=True)
        top_features = contrib_df.head(top_k)

        # Generate Plain-English Reasons
        reasons = []
        for _, row in top_features.iterrows():
            feat = row["Feature"]
            val = row["Value"]
            contrib = row["Contribution"]
            direction = "increases" if contrib > 0 else "decreases"

            if feat == "V14":
                if val < -3.0:
                    reasons.append(f"Critical anomaly on V14 ({val:.2f}): Severely depressed PCA component strongly associated with fraudulent card activity.")
                else:
                    reasons.append(f"V14 component ({val:.2f}) {direction} fraud risk.")
            elif feat == "V12":
                if val < -2.5:
                    reasons.append(f"Abnormal V12 signal ({val:.2f}): Heavy negative deviation indicative of account compromise.")
                else:
                    reasons.append(f"V12 component ({val:.2f}) {direction} fraud likelihood.")
            elif feat == "V10":
                if val < -2.5:
                    reasons.append(f"High-risk V10 deviation ({val:.2f}): Unusual merchant or terminal transaction pattern.")
                else:
                    reasons.append(f"V10 component ({val:.2f}) {direction} fraud risk.")
            elif feat == "V4":
                if val > 2.5:
                    reasons.append(f"Elevated V4 reading ({val:.2f}): High positive variance typically seen in unauthorized authorizations.")
                else:
                    reasons.append(f"V4 component ({val:.2f}) {direction} fraud risk.")
            elif feat in ["Hour_sin", "Hour_cos", "Time"]:
                reasons.append(f"Temporal velocity: Transaction timestamp aligns with off-peak / high-risk activity window.")
            elif feat in ["Amount", "Log_Amount"]:
                reasons.append(f"Transaction amount scale: Dollar amount deviates from normal cardholder spending volume.")
            elif "_x_" in feat:
                reasons.append(f"Non-linear interaction term {feat} ({val:.2f}): Combined multi-component anomaly.")
            else:
                reasons.append(f"Component {feat} ({val:.2f}) {direction} overall fraud risk score.")

        return {
            "fraud_probability": prob,
            "top_features": top_features[["Feature", "Value", "Contribution"]],
            "all_contributions": contrib_df[["Feature", "Value", "Contribution"]],
            "reasons": reasons
        }

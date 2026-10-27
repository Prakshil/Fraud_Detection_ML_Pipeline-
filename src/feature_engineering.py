import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Production-Grade Hybrid Feature Engineering Transformer.
    Combines:
    1. Robust scaling on skewed transaction Amount and elapsed Time (Notebook 1).
    2. Cyclical transformation of time-of-day (Notebook 3 domain insight).
    3. Log-transformation of Amount to compress extreme positive skew (0 to $25k+).
    4. Non-linear interaction features for highest-discriminating PCA components (Notebooks 1, 2, 3).
    5. Anomaly tail indicator flags for high-risk component boundaries.
    """
    def __init__(self, add_interactions=True, add_cyclical_time=True, add_tail_flags=True):
        self.add_interactions = add_interactions
        self.add_cyclical_time = add_cyclical_time
        self.add_tail_flags = add_tail_flags
        self.scaler = RobustScaler()
        self.scale_cols = ['Time', 'Amount', 'Log_Amount']
        self.feature_names_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_copy = X.copy()
        if 'Amount' in X_copy.columns:
            X_copy['Log_Amount'] = np.log1p(np.maximum(0, X_copy['Amount']))
        else:
            X_copy['Log_Amount'] = 0.0
            
        fit_cols = [c for c in self.scale_cols if c in X_copy.columns]
        if fit_cols:
            self.scaler.fit(X_copy[fit_cols])
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        df = X.copy()
        
        # 1. Log Amount
        if 'Amount' in df.columns:
            df['Log_Amount'] = np.log1p(np.maximum(0, df['Amount']))
        
        # 2. Robust Scaling
        scale_cols = [c for c in self.scale_cols if c in df.columns]
        if scale_cols:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        # 3. Cyclical Time Features (24-hour periodicity)
        if self.add_cyclical_time and 'Time' in df.columns:
            # Assuming Time is in seconds from start
            hour = (df['Time'] // 3600) % 24
            df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
        # 4. Critical PCA Interaction Terms (from top correlated components V14, V12, V10, V17, V4, V11)
        if self.add_interactions:
            if 'V14' in df.columns and 'V12' in df.columns:
                df['V14_x_V12'] = df['V14'] * df['V12']
            if 'V14' in df.columns and 'V17' in df.columns:
                df['V14_x_V17'] = df['V14'] * df['V17']
            if 'V10' in df.columns and 'V11' in df.columns:
                df['V10_x_V11'] = df['V10'] * df['V11']
            if 'V4' in df.columns and 'V11' in df.columns:
                df['V4_x_V11'] = df['V4'] * df['V11']
                
        # 5. Tail Indicator Flags (Anomalous extremes from Notebook 3 insight)
        if self.add_tail_flags:
            if 'V14' in df.columns:
                df['V14_extreme_neg'] = (df['V14'] < -3.0).astype(float)
            if 'V12' in df.columns:
                df['V12_extreme_neg'] = (df['V12'] < -2.5).astype(float)
            if 'V10' in df.columns:
                df['V10_extreme_neg'] = (df['V10'] < -2.5).astype(float)
            if 'V4' in df.columns:
                df['V4_extreme_pos'] = (df['V4'] > 2.5).astype(float)
        
        self.feature_names_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

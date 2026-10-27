import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Production-Grade Hybrid Feature Engineering Transformer.
    Combines:
    1. Cyclical transformation of time-of-day (24-hour periodicity) using raw Time.
    2. Log-transformation of Amount to compress extreme positive skew (0 to $25k+).
    3. Robust scaling on skewed transaction Amount, elapsed Time, and Log_Amount.
    4. Non-linear interaction features for highest-discriminating PCA components.
    5. Anomaly tail indicator flags for high-risk component boundaries.
    """
    def __init__(self, add_interactions=True, add_cyclical_time=True, add_tail_flags=True):
        self.add_interactions = add_interactions
        self.add_cyclical_time = add_cyclical_time
        self.add_tail_flags = add_tail_flags
        self.scaler = RobustScaler()
        self.scale_cols = ['Time', 'Amount', 'Log_Amount']
        self.feature_names_in_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_df = X.copy()
        else:
            X_arr = np.asarray(X)
            if self.feature_names_in_ is not None and X_arr.shape[1] == len(self.feature_names_in_):
                cols = self.feature_names_in_
            elif X_arr.shape[1] == 30:
                cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
                self.feature_names_in_ = cols
            else:
                cols = [f'feat_{i}' for i in range(X_arr.shape[1])]
                self.feature_names_in_ = cols
            X_df = pd.DataFrame(X_arr, columns=cols)
        
        if 'Amount' in X_df.columns:
            X_df['Log_Amount'] = np.log1p(np.maximum(0, X_df['Amount']))
        else:
            X_df['Log_Amount'] = 0.0
            
        fit_cols = [c for c in self.scale_cols if c in X_df.columns]
        if fit_cols:
            self.scaler.fit(X_df[fit_cols])
            
        # Determine output feature names reliably
        sample_transformed = self.transform(X_df.iloc[:2])
        self.feature_names_ = list(sample_transformed.columns)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            X_arr = np.asarray(X)
            cols = self.feature_names_in_ if self.feature_names_in_ is not None and X_arr.shape[1] == len(self.feature_names_in_) else None
            if cols is None and X_arr.shape[1] == 30:
                cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            elif cols is None:
                cols = [f'feat_{i}' for i in range(X_arr.shape[1])]
            df = pd.DataFrame(X_arr, columns=cols)
        
        # 1. Cyclical Time Features (Calculated from raw Time in seconds before scaling)
        if self.add_cyclical_time and 'Time' in df.columns:
            hour = (df['Time'] // 3600) % 24
            df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
        # 2. Log Amount
        if 'Amount' in df.columns:
            df['Log_Amount'] = np.log1p(np.maximum(0, df['Amount']))
        
        # 3. Robust Scaling (Applied to raw Time, Amount, and Log_Amount)
        scale_cols = [c for c in self.scale_cols if c in df.columns]
        if scale_cols:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
            
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
                
        # 5. Tail Indicator Flags (Anomalous extremes from domain insight)
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
        return np.array(self.feature_names_, dtype=object)

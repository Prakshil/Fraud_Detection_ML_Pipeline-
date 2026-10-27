import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import FraudFeatureEngineer

@pytest.fixture
def sample_transaction_df():
    np.random.seed(42)
    n = 200
    # Create time spanning 48 hours (0 to 172800 seconds)
    times = np.linspace(0, 172800, n)
    amounts = np.random.exponential(scale=50, size=n)
    v_features = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data = {"Time": times, **v_features, "Amount": amounts}
    return pd.DataFrame(data)

def test_cyclical_time_generation(sample_transaction_df):
    """Verifies that cyclical time produces full diurnal coverage and is not degenerate."""
    fe = FraudFeatureEngineer(add_cyclical_time=True)
    transformed = fe.fit_transform(sample_transaction_df)

    assert "Hour_sin" in transformed.columns
    assert "Hour_cos" in transformed.columns
    # Must have multiple unique values across 24 hours, not just 2 degenerate values
    assert len(transformed["Hour_sin"].unique()) > 10
    assert len(transformed["Hour_cos"].unique()) > 10
    assert transformed["Hour_sin"].min() >= -1.0
    assert transformed["Hour_sin"].max() <= 1.0

def test_numpy_array_input_compatibility(sample_transaction_df):
    """Verifies that FraudFeatureEngineer gracefully accepts NumPy arrays without shape mismatch."""
    fe = FraudFeatureEngineer()
    df_result = fe.fit_transform(sample_transaction_df)

    # Pass as raw NumPy array
    arr_result = fe.transform(sample_transaction_df.values)

    assert arr_result.shape == df_result.shape
    assert list(arr_result.columns) == list(df_result.columns)

def test_pca_interaction_terms(sample_transaction_df):
    """Verifies that interaction terms are correctly computed."""
    fe = FraudFeatureEngineer(add_interactions=True)
    transformed = fe.fit_transform(sample_transaction_df)

    for col in ["V14_x_V12", "V14_x_V17", "V10_x_V11", "V4_x_V11"]:
        assert col in transformed.columns

    # Verify mathematical accuracy of interaction term
    expected_v14_v12 = sample_transaction_df["V14"] * sample_transaction_df["V12"]
    np.testing.assert_allclose(transformed["V14_x_V12"], expected_v14_v12)

def test_get_feature_names_out(sample_transaction_df):
    """Verifies standard scikit-learn get_feature_names_out API."""
    fe = FraudFeatureEngineer()
    fe.fit(sample_transaction_df)
    names = fe.get_feature_names_out()

    assert isinstance(names, np.ndarray)
    assert len(names) == 41

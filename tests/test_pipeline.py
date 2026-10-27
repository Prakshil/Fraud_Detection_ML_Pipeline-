import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from src.pipeline import FraudDetectionPipeline

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 300
    times = np.linspace(0, 100000, n)
    amounts = np.random.uniform(5, 300, n)
    v_cols = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    df = pd.DataFrame({"Time": times, **v_cols, "Amount": amounts})
    # Imbalanced labels: ~5% fraud
    y = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    return df, y

def test_pipeline_fit_and_predict(synthetic_data):
    X, y = synthetic_data
    pipe = FraudDetectionPipeline(model_type="logreg", max_iter=200)
    pipe.fit(X, y, tune_thresh=True)

    assert pipe.is_fitted
    assert 0.0 <= pipe.threshold <= 1.0

    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert len(preds) == len(X)
    assert len(probs) == len(X)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert set(np.unique(preds)).issubset({0, 1})

def test_pipeline_numpy_input(synthetic_data):
    """Verifies that the entire pipeline can predict on raw NumPy array input."""
    X, y = synthetic_data
    pipe = FraudDetectionPipeline(model_type="logreg", max_iter=200)
    pipe.fit(X, y, tune_thresh=False)

    preds_arr = pipe.predict(X.values)
    probs_arr = pipe.predict_proba(X.values)

    assert len(preds_arr) == len(X)
    assert len(probs_arr) == len(X)

def test_pipeline_serialization_roundtrip(synthetic_data):
    X, y = synthetic_data
    pipe = FraudDetectionPipeline(model_type="logreg", max_iter=200)
    pipe.fit(X, y, tune_thresh=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_pipeline.joblib")
        pipe.save(save_path)
        assert os.path.exists(save_path)

        reloaded = FraudDetectionPipeline.load(save_path)
        assert reloaded.is_fitted
        assert reloaded.threshold == pipe.threshold

        orig_preds = pipe.predict(X)
        reloaded_preds = reloaded.predict(X)
        np.testing.assert_array_equal(orig_preds, reloaded_preds)

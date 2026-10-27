import pytest
import numpy as np
import pandas as pd
from src.pipeline import FraudDetectionPipeline
from src.explainer import FraudExplainer

@pytest.fixture
def trained_rf_pipeline():
    np.random.seed(42)
    n = 200
    times = np.linspace(0, 100000, n)
    amounts = np.random.uniform(5, 300, n)
    v_cols = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    df = pd.DataFrame({"Time": times, **v_cols, "Amount": amounts})
    y = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

    pipe = FraudDetectionPipeline(model_type="rf", n_estimators=10, max_depth=5)
    pipe.fit(df, y, tune_thresh=False)
    return pipe, df

def test_fraud_explainer_rf(trained_rf_pipeline):
    pipe, df = trained_rf_pipeline
    explainer = FraudExplainer(pipe)

    # Explain single row
    sample_tx = df.iloc[[0]]
    result = explainer.explain_transaction(sample_tx, top_k=5)

    assert "fraud_probability" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert "top_features" in result
    assert len(result["top_features"]) <= 5
    assert "reasons" in result
    assert len(result["reasons"]) > 0
    assert isinstance(result["reasons"][0], str)

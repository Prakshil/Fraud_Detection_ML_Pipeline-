import numpy as np
import pytest
from src.evaluation import find_optimal_threshold, evaluate_model

@pytest.fixture
def dummy_predictions():
    np.random.seed(42)
    n = 1000
    # 5% fraud prevalence
    y_true = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    # Give higher probabilities to true positives
    y_probs = np.where(y_true == 1, np.random.beta(5, 2, size=n), np.random.beta(1, 5, size=n))
    return y_true, y_probs

def test_find_optimal_threshold_f1(dummy_predictions):
    y_true, y_probs = dummy_predictions
    best_thresh, best_metric, metric_name = find_optimal_threshold(y_true, y_probs, metric="f1")

    assert 0.0 <= best_thresh <= 1.0
    assert 0.0 <= best_metric <= 1.0
    assert metric_name == "f1"

def test_find_optimal_threshold_cost(dummy_predictions):
    y_true, y_probs = dummy_predictions
    best_thresh, best_cost, metric_name = find_optimal_threshold(
        y_true, y_probs, metric="cost", cost_fn=122.21, cost_fp=10.0
    )

    assert 0.0 <= best_thresh <= 1.0
    assert best_cost >= 0.0
    assert metric_name == "cost"

def test_evaluate_model(dummy_predictions):
    y_true, y_probs = dummy_predictions
    res = evaluate_model(y_true, y_probs, threshold=0.5, model_name="Test-Model")

    expected_keys = [
        "model_name", "threshold", "pr_auc", "roc_auc", "precision",
        "recall", "f1_score", "confusion_matrix", "true_positives",
        "false_positives", "true_negatives", "false_negatives"
    ]
    for key in expected_keys:
        assert key in res

    assert res["true_positives"] + res["false_negatives"] == np.sum(y_true == 1)
    assert res["true_negatives"] + res["false_positives"] == np.sum(y_true == 0)

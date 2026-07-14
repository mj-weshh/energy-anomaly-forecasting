import numpy as np
import pytest

from src.models.evaluate_models import evaluate_anomaly_model


def test_evaluate_anomaly_model_returns_expected_keys() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    metrics = evaluate_anomaly_model(y_true, y_pred)
    assert set(metrics) == {"precision", "recall", "f1", "confusion_matrix"}
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_evaluate_anomaly_model_rejects_non_binary() -> None:
    with pytest.raises(ValueError):
        evaluate_anomaly_model(np.array([0, 1, 2]), np.array([0, 1, 0]))

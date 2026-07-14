import numpy as np
import pandas as pd
import pytest

from src.models.tuning_utils import align_labels


def test_align_labels_maps_normal_and_abnormal() -> None:
    df = pd.DataFrame(
        {
            "Anomaly_Label": ["Normal", "Abnormal", "Normal"],
        }
    )
    y = align_labels(df, df.index)
    np.testing.assert_array_equal(y, np.array([0, 1, 0], dtype=int))


def test_align_labels_raises_on_missing_column() -> None:
    df = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(KeyError, match="Anomaly_Label"):
        align_labels(df, df.index)


def test_align_labels_raises_on_unknown_value() -> None:
    df = pd.DataFrame({"Anomaly_Label": ["Normal", "Unknown"]})
    with pytest.raises(ValueError, match="Unmapped"):
        align_labels(df, df.index)

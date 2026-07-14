import numpy as np

from src.models.tuning_utils import temporal_train_val_test_split


def test_temporal_split_sizes_on_4953_rows() -> None:
    train_idx, val_idx, test_idx = temporal_train_val_test_split(4953)
    assert len(train_idx) == 2971
    assert len(val_idx) == 991
    assert len(test_idx) == 991
    assert train_idx[-1] + 1 == val_idx[0]
    assert val_idx[-1] + 1 == test_idx[0]
    assert np.array_equal(np.concatenate([train_idx, val_idx, test_idx]), np.arange(4953))

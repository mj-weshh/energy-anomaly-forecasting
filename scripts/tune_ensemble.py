"""Compare IF, DBSCAN, and ensemble strategies on validation; report test F1.

Uses tuned configs from ``src/models/anomaly_config.py`` (run ``tune_isolation_forest.py``
and ``tune_dbscan.py`` first to refresh).

Run from repository root::

    python scripts/tune_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import (  # noqa: E402
    find_dataset_csv,
    get_project_root,
    load_smart_meter_data,
)
from src.features.build_features import build_enhanced_anomaly_features  # noqa: E402
from src.models.anomaly_config import (  # noqa: E402
    BEST_DBSCAN_CONFIG,
    BEST_ENSEMBLE_CONFIG,
    BEST_IF_CONFIG,
)
from src.models.anomaly_preprocessing import AnomalyPreprocessor  # noqa: E402
from src.models.evaluate_models import evaluate_anomaly_model, evaluate_on_splits  # noqa: E402
from src.models.train_anomaly_models import prepare_feature_matrix  # noqa: E402
from src.models.tuning_utils import (  # noqa: E402
    align_labels,
    isolation_forest_scores,
    normalize_scores,
    predict_from_scores,
    temporal_train_val_test_split,
)

STRATEGIES = ["intersection", "union", "weighted"]
ALPHA_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]


def _prepare(df):
    feature_matrix = prepare_feature_matrix(df)
    y_true = align_labels(df, feature_matrix.index)
    train_idx, val_idx, test_idx = temporal_train_val_test_split(len(feature_matrix))
    hours = df.loc[feature_matrix.index, "hour"]
    consumption = df.loc[feature_matrix.index, "Electricity_Consumed"]
    preprocessor = AnomalyPreprocessor()
    train_mask = np.zeros(len(feature_matrix), dtype=bool)
    train_mask[train_idx] = True
    preprocessor.fit(feature_matrix, hours, consumption, train_mask)
    X = preprocessor.transform(feature_matrix, hours, consumption)
    return X, y_true, train_idx, val_idx, test_idx


def main() -> None:
    csv_path = find_dataset_csv(get_project_root())
    df = build_enhanced_anomaly_features(load_smart_meter_data(csv_path))
    X, y_true, train_idx, val_idx, test_idx = _prepare(df)

    if_model = IsolationForest(
        contamination=BEST_IF_CONFIG["contamination"],
        n_estimators=BEST_IF_CONFIG["n_estimators"],
        max_features=BEST_IF_CONFIG["max_features"],
        random_state=42,
    )
    if_model.fit(X[train_idx])
    threshold = float(BEST_IF_CONFIG["score_threshold"])
    if_scores = isolation_forest_scores(if_model, X)
    if_preds = predict_from_scores(if_scores, threshold)

    db_preds = (
        DBSCAN(
            eps=BEST_DBSCAN_CONFIG["eps"],
            min_samples=BEST_DBSCAN_CONFIG["min_samples"],
            metric=BEST_DBSCAN_CONFIG["metric"],
        ).fit_predict(X)
        == -1
    ).astype(int)

    best_val_f1 = -1.0
    best_result: dict = {}

    print(f"Loaded: {csv_path}")
    print(
        f"IF: contamination={BEST_IF_CONFIG['contamination']}, "
        f"n_estimators={BEST_IF_CONFIG['n_estimators']}, threshold={threshold:.4f}"
    )
    print(
        f"DBSCAN: eps={BEST_DBSCAN_CONFIG['eps']}, "
        f"min_samples={BEST_DBSCAN_CONFIG['min_samples']}, "
        f"metric={BEST_DBSCAN_CONFIG['metric']}\n"
    )
    print(f"{'Strategy':<14} {'Alpha':>6} {'Val F1':>8} {'Test F1':>8}")
    print("-" * 40)

    for strategy in STRATEGIES:
        alphas = ALPHA_VALUES if strategy == "weighted" else [BEST_ENSEMBLE_CONFIG["alpha"]]
        for alpha in alphas:
            if strategy == "union":
                preds = np.maximum(if_preds, db_preds)
            elif strategy == "intersection":
                preds = np.minimum(if_preds, db_preds)
            else:
                combined = alpha * normalize_scores(if_scores) + (1 - alpha) * db_preds
                preds = (combined >= 0.5).astype(int)

            val_m = evaluate_anomaly_model(y_true[val_idx], preds[val_idx])
            test_m = evaluate_anomaly_model(y_true[test_idx], preds[test_idx])
            alpha_label = f"{alpha:.1f}" if strategy == "weighted" else "-"
            print(
                f"{strategy:<14} {alpha_label:>6} {val_m['f1']:8.3f} {test_m['f1']:8.3f}"
            )
            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                best_result = {"strategy": strategy, "alpha": alpha, "preds": preds}

    split_metrics = evaluate_on_splits(
        y_true, best_result["preds"], train_idx, val_idx, test_idx
    )
    print("\nBest ensemble on validation:")
    print(f"  strategy={best_result['strategy']}  alpha={best_result.get('alpha', '-')}")
    for split_name, metrics in split_metrics.items():
        print(f"  {split_name}: F1={metrics['f1']:.3f}")


if __name__ == "__main__":
    main()

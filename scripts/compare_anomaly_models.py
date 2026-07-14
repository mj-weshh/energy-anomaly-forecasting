"""Research dashboard: legacy vs enhanced anomaly detectors with temporal splits.

Run from repository root::

    python scripts/compare_anomaly_models.py
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
from src.features.build_features import (  # noqa: E402
    build_all_features,
    build_enhanced_anomaly_features,
)
from src.models.evaluate_models import evaluate_anomaly_model, evaluate_on_splits  # noqa: E402
from src.models.train_anomaly_models import train_dbscan, train_isolation_forest  # noqa: E402
from src.models.feature_matrix import prepare_feature_matrix  # noqa: E402
from src.models.tuning_utils import (  # noqa: E402
    align_labels,
    find_best_threshold,
    isolation_forest_scores,
    predict_from_scores,
    prepare_temporal_tuning_data,
    temporal_train_val_test_split,
)

from src.models.anomaly_config import (  # noqa: E402
    BEST_DBSCAN_CONFIG,
    BEST_IF_CONFIG,
    TUNING_METRICS,
)


def _print_row(label: str, metrics: dict) -> None:
    print(
        f"{label:<28} {metrics['precision']:8.3f} "
        f"{metrics['recall']:8.3f} {metrics['f1']:8.3f}"
    )


def _print_confusion_matrix(label: str, metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print(f"\n{label} [[TN, FP], [FN, TP]]:")
    print(f"  TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}")
    print(f"  FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")


def main() -> None:
    csv_path = find_dataset_csv(get_project_root())
    raw = load_smart_meter_data(csv_path)

    print(f"Loaded: {csv_path}\n")
    print(f"{'Model':<28} {'Precision':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 56)

    # Legacy IF — full dataset (production baseline)
    df_legacy = build_all_features(raw)
    idx_legacy = prepare_feature_matrix(df_legacy).index
    y_legacy = align_labels(df_legacy, idx_legacy)
    _, legacy_preds = train_isolation_forest(df_legacy)
    legacy_metrics = evaluate_anomaly_model(y_legacy, legacy_preds)
    _print_row("Legacy IF (full data)", legacy_metrics)

    # Shared temporal split (4953 eval rows, chronological 60/20/20)
    legacy_matrix = prepare_feature_matrix(df_legacy)
    legacy_X = legacy_matrix.to_numpy(dtype=float)
    train_idx, val_idx, test_idx = temporal_train_val_test_split(len(legacy_matrix))

    legacy_if_model = IsolationForest(
        contamination=0.05,
        n_estimators=100,
        max_features=1.0,
        random_state=42,
    )
    legacy_if_model.fit(legacy_X[train_idx])
    legacy_test_preds = (legacy_if_model.predict(legacy_X) == -1).astype(int)
    legacy_test_metrics = evaluate_anomaly_model(
        y_legacy[test_idx], legacy_test_preds[test_idx]
    )
    _print_row("Legacy IF (test)", legacy_test_metrics)

    legacy_val_scores = isolation_forest_scores(legacy_if_model, legacy_X[val_idx])
    legacy_threshold, _ = find_best_threshold(legacy_val_scores, y_legacy[val_idx])
    legacy_thresh_preds = predict_from_scores(
        isolation_forest_scores(legacy_if_model, legacy_X), legacy_threshold
    )
    legacy_thresh_test_metrics = evaluate_anomaly_model(
        y_legacy[test_idx], legacy_thresh_preds[test_idx]
    )
    _print_row("Legacy IF (test, val threshold)", legacy_thresh_test_metrics)

    # Enhanced pipelines with temporal test split
    df_enh = build_enhanced_anomaly_features(raw)
    enh = prepare_temporal_tuning_data(df_enh, scale=True)
    train_idx, val_idx, test_idx = enh.train_idx, enh.val_idx, enh.test_idx
    y_true, X = enh.y_true, enh.X

    # Enhanced IF — train on train, threshold on val, report test
    if_model = IsolationForest(
        contamination=BEST_IF_CONFIG["contamination"],
        n_estimators=BEST_IF_CONFIG["n_estimators"],
        max_features=BEST_IF_CONFIG["max_features"],
        random_state=42,
    )
    if_model.fit(X[train_idx])
    threshold = BEST_IF_CONFIG.get("score_threshold")
    if threshold is None:
        val_scores = isolation_forest_scores(if_model, X[val_idx])
        threshold, _ = find_best_threshold(val_scores, y_true[val_idx])
    if_preds = predict_from_scores(isolation_forest_scores(if_model, X), float(threshold))
    if_test = evaluate_anomaly_model(y_true[test_idx], if_preds[test_idx])
    _print_row("Enhanced IF (test)", if_test)

    # Enhanced DBSCAN — scaled
    db_preds = (
        DBSCAN(
            eps=BEST_DBSCAN_CONFIG["eps"],
            min_samples=BEST_DBSCAN_CONFIG["min_samples"],
            metric=BEST_DBSCAN_CONFIG["metric"],
        ).fit_predict(X)
        == -1
    ).astype(int)
    db_test = evaluate_anomaly_model(y_true[test_idx], db_preds[test_idx])
    _print_row("Enhanced DBSCAN (test)", db_test)

    # Enhanced ensemble — union and intersection
    ens_union_preds = np.maximum(if_preds, db_preds)
    ens_union_test = evaluate_anomaly_model(y_true[test_idx], ens_union_preds[test_idx])
    _print_row("Enhanced ensemble union (test)", ens_union_test)

    ens_intersect_preds = np.minimum(if_preds, db_preds)
    ens_intersect_test = evaluate_anomaly_model(y_true[test_idx], ens_intersect_preds[test_idx])
    _print_row("Enhanced ensemble intersect (test)", ens_intersect_test)

    # Legacy DBSCAN for reference
    _, legacy_db = train_dbscan(df_legacy, eps=0.5, min_samples=5, scale=False)
    legacy_db_metrics = evaluate_anomaly_model(y_legacy, legacy_db)
    _print_row("Legacy DBSCAN (full data)", legacy_db_metrics)

    print("\nEnhanced IF train/val/test breakdown:")
    split_metrics = evaluate_on_splits(y_true, if_preds, train_idx, val_idx, test_idx)
    for split_name, metrics in split_metrics.items():
        print(f"  {split_name}: F1={metrics['f1']:.3f}")

    print("\nFair head-to-head — test split confusion matrices:")
    _print_confusion_matrix("Legacy IF (test, production params)", legacy_test_metrics)
    _print_confusion_matrix(
        "Legacy IF (test, val threshold)", legacy_thresh_test_metrics
    )
    _print_confusion_matrix("Enhanced IF (test)", if_test)

    print("\nReference tuning metrics (test split, from anomaly_config):")
    print(f"  Enhanced IF:        F1={TUNING_METRICS['enhanced_if_test_f1']:.3f}")
    print(f"  Enhanced DBSCAN:    F1={TUNING_METRICS['enhanced_dbscan_test_f1']:.3f}")
    print(
        "  Ensemble union:     "
        f"F1={TUNING_METRICS['enhanced_ensemble_union_test_f1']:.3f}"
    )
    print(
        "  Ensemble intersect: "
        f"F1={TUNING_METRICS['enhanced_ensemble_intersection_test_f1']:.3f}"
    )


if __name__ == "__main__":
    main()

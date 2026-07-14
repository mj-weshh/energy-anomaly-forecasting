"""Compare IF, DBSCAN, and ensemble strategies on validation; report test F1.

Uses tuned configs from ``src/models/anomaly_config.py``.

Run from repository root::

    python scripts/tune_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ingest_data import find_dataset_csv, get_project_root, load_smart_meter_data  # noqa: E402
from src.features.build_features import build_enhanced_anomaly_features  # noqa: E402
from src.models.anomaly_config import BEST_DBSCAN_CONFIG, BEST_ENSEMBLE_CONFIG, BEST_IF_CONFIG  # noqa: E402
from src.models.evaluate_models import evaluate_anomaly_model, evaluate_on_splits  # noqa: E402
from src.models.train_anomaly_models import train_ensemble  # noqa: E402
from src.models.tuning_utils import prepare_temporal_tuning_data  # noqa: E402

STRATEGIES = ["intersection", "union", "weighted"]
ALPHA_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> None:
    csv_path = find_dataset_csv(get_project_root())
    df = build_enhanced_anomaly_features(load_smart_meter_data(csv_path))
    data = prepare_temporal_tuning_data(df, scale=True)

    if_kwargs = {
        "contamination": BEST_IF_CONFIG["contamination"],
        "n_estimators": BEST_IF_CONFIG["n_estimators"],
        "max_features": BEST_IF_CONFIG["max_features"],
        "scale": True,
        "score_threshold": float(BEST_IF_CONFIG["score_threshold"]),
    }
    dbscan_kwargs = {
        "eps": BEST_DBSCAN_CONFIG["eps"],
        "min_samples": BEST_DBSCAN_CONFIG["min_samples"],
        "metric": BEST_DBSCAN_CONFIG["metric"],
        "scale": True,
    }

    best_val_f1 = -1.0
    best_result: dict = {}

    print(f"Loaded: {csv_path}")
    print(
        f"IF: contamination={BEST_IF_CONFIG['contamination']}, "
        f"n_estimators={BEST_IF_CONFIG['n_estimators']}, "
        f"threshold={BEST_IF_CONFIG['score_threshold']:.4f}"
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
            _, preds = train_ensemble(
                df,
                if_kwargs=if_kwargs,
                dbscan_kwargs=dbscan_kwargs,
                strategy=strategy,
                alpha=alpha,
                score_threshold=float(BEST_IF_CONFIG["score_threshold"]),
                fit_indices=data.train_idx,
            )
            val_m = evaluate_anomaly_model(data.y_true[data.val_idx], preds[data.val_idx])
            test_m = evaluate_anomaly_model(data.y_true[data.test_idx], preds[data.test_idx])
            alpha_label = f"{alpha:.1f}" if strategy == "weighted" else "-"
            print(
                f"{strategy:<14} {alpha_label:>6} {val_m['f1']:8.3f} {test_m['f1']:8.3f}"
            )
            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                best_result = {"strategy": strategy, "alpha": alpha, "preds": preds}

    split_metrics = evaluate_on_splits(
        data.y_true, best_result["preds"], data.train_idx, data.val_idx, data.test_idx
    )
    print("\nBest ensemble on validation:")
    print(f"  strategy={best_result['strategy']}  alpha={best_result.get('alpha', '-')}")
    for split_name, metrics in split_metrics.items():
        print(f"  {split_name}: F1={metrics['f1']:.3f}")


if __name__ == "__main__":
    main()

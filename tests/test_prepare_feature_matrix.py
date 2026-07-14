from src.data.ingest_data import find_dataset_csv, get_project_root, load_smart_meter_data
from src.features.build_features import build_all_features
from src.models.feature_matrix import prepare_feature_matrix


def test_prepare_feature_matrix_drops_warm_up_rows() -> None:
    csv_path = find_dataset_csv(get_project_root())
    df = build_all_features(load_smart_meter_data(csv_path))
    matrix = prepare_feature_matrix(df)
    assert len(matrix) == 4953
    assert matrix.isna().sum().sum() == 0

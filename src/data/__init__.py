"""Data loading and validation utilities for smart meter time-series data.

Public API is defined in :mod:`src.data.ingest_data`. Import directly::

    from src.data.ingest_data import load_smart_meter_data, find_dataset_csv
"""

__all__ = [
    "check_time_continuity",
    "find_dataset_csv",
    "get_project_root",
    "load_smart_meter_data",
    "print_schema_summary",
]

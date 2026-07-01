"""Generate architecture flowchart PNGs for MkDocs via mermaid.ink.

Run from repository root::

    python scripts/generate_mermaid_assets.py
"""

from __future__ import annotations

import base64
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs" / "assets"
MERMAID_INK_BASE = "https://mermaid.ink/img/"

SYSTEM_OVERVIEW = """\
flowchart LR
    kaggle[KaggleDataset] --> rawCsv[smart_meter_data.csv]
    rawCsv --> ingest[src/data/ingest_data.py]
    ingest --> validatedDF[ValidatedDataFrame]
    validatedDF --> visualize[src/visualization/visualize.py]
    validatedDF --> eda[Phase1_EDA]
    validatedDF --> anomaly[Phase2_AnomalyDetection]
    validatedDF --> forecast[Phase3_Forecasting]
    visualize --> docAssets[docs/assets/eda]
"""

INGESTION_PIPELINE = """\
flowchart TD
    start[main] --> findCsv[find_dataset_csv]
    findCsv --> load[load_smart_meter_data]
    load --> parseTimestamp[Parse Timestamp to datetime64]
    parseTimestamp --> sortRows[Sort chronologically]
    sortRows --> schema[print_schema_summary]
    schema --> continuity[check_time_continuity]
    continuity --> done[Report PASS or REVIEW]
"""

DIAGRAMS: dict[str, str] = {
    "system-overview.png": SYSTEM_OVERVIEW,
    "ingestion-pipeline.png": INGESTION_PIPELINE,
}


def _encode_diagram(diagram: str) -> str:
    return base64.urlsafe_b64encode(diagram.encode("utf-8")).decode("ascii")


def _fetch_png(diagram: str) -> bytes:
    url = f"{MERMAID_INK_BASE}{_encode_diagram(diagram)}"
    request = urllib.request.Request(url, headers={"User-Agent": "energy-anomaly-forecasting/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for filename, diagram in DIAGRAMS.items():
        path = OUTPUT_DIR / filename
        try:
            png_bytes = _fetch_png(diagram)
        except urllib.error.URLError as exc:
            print(f"Failed to fetch {filename}: {exc}", file=sys.stderr)
            sys.exit(1)
        path.write_bytes(png_bytes)
        saved.append(path)

    print(f"Exported {len(saved)} PNGs to {OUTPUT_DIR}:")
    for path in saved:
        print(f"  - {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

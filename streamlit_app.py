"""
Ponto de entrada para deploy (Streamlit Community Cloud).

Usa artefatos pré-gerados em data/processed_demo/ (dados sintéticos).
O pipeline completo não roda na nuvem — evita timeout e arquivos .joblib corrompidos.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("COSMETICOS_RAW_DATA_DIR", str(ROOT / "data" / "raw_fake"))
os.environ.setdefault("COSMETICOS_PROCESSED_DATA_DIR", str(ROOT / "data" / "processed_demo"))

_DEMO_DIR = ROOT / "data" / "processed_demo"
_REQUIRED = (
    "campanha_top100.csv",
    "kpis_diarios.parquet",
    "propensity_metrics.csv",
    "forecast_future_30d.csv",
)

_missing = [f for f in _REQUIRED if not (_DEMO_DIR / f).exists()]
if _missing:
    raise FileNotFoundError(
        f"Artefatos de demo ausentes em data/processed_demo/: {_missing}. "
        "Rode: python scripts/sync_processed_demo.py"
    )

_dashboard_path = ROOT / "src" / "cosmeticos_ia" / "app" / "dashboard.py"
_spec = importlib.util.spec_from_file_location("cosmeticos_dashboard", _dashboard_path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

"""
Ponto de entrada para deploy (Streamlit Community Cloud).

- Configura PYTHONPATH
- Gera artefatos com dados sintéticos se data/processed/ estiver vazio
- Carrega o dashboard executivo
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("COSMETICOS_RAW_DATA_DIR", str(ROOT / "data" / "raw_fake"))

from cosmeticos_ia.config import PROCESSED_DATA_DIR  # noqa: E402

_MARKER = PROCESSED_DATA_DIR / "campanha_top100.csv"


def _ensure_artifacts() -> None:
    if _MARKER.exists():
        return
    from cosmeticos_ia.pipelines.run_all_pipelines import main as run_pipeline

    run_pipeline()


_ensure_artifacts()

_dashboard_path = ROOT / "src" / "cosmeticos_ia" / "app" / "dashboard.py"
_spec = importlib.util.spec_from_file_location("cosmeticos_dashboard", _dashboard_path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

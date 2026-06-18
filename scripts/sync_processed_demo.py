"""Copia artefatos do dashboard (data/processed) para data/processed_demo (deploy)."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed"
DST = ROOT / "data" / "processed_demo"

DASHBOARD_ARTIFACTS = [
    "campanha_top100.csv",
    "kpis_diarios.parquet",
    "propensity_metrics.csv",
    "forecast_metrics.csv",
    "forecast_backtest_summary.csv",
    "forecast_future_30d.csv",
    "forecast_predictions_test.csv",
    "propensity_recall_at_k_by_snapshot.csv",
    "campaign_uplift_summary.csv",
    "campaign_uplift_vs_baseline.csv",
    "monitoring_report.csv",
    "monitoring_summary.csv",
]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in DASHBOARD_ARTIFACTS:
        src = SRC / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, DST / name)
        print(f"  {name}")

    if missing:
        raise SystemExit(
            f"Artefatos ausentes em {SRC}: {missing}\n"
            "Rode o pipeline antes: python -m cosmeticos_ia.pipelines.run_all_pipelines"
        )
    print(f"\n{len(DASHBOARD_ARTIFACTS)} artefatos sincronizados em {DST}")


if __name__ == "__main__":
    main()

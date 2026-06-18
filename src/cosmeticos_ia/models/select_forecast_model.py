from __future__ import annotations

import shutil

import pandas as pd

from cosmeticos_ia.config import PROCESSED_DATA_DIR

# Apenas modelos supervisionados possuem artefato .joblib para produção
PRODUCTION_CANDIDATES = {
    "random_forest": PROCESSED_DATA_DIR / "forecast_model_random_forest.joblib",
    "xgboost": PROCESSED_DATA_DIR / "forecast_model_xgboost.joblib",
}


def main() -> None:
    summary_path = PROCESSED_DATA_DIR / "forecast_backtest_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Resumo de backtest não encontrado: {summary_path}")

    summary = pd.read_csv(summary_path)
    eligible = summary[summary["model"].isin(PRODUCTION_CANDIDATES)].sort_values("wape_mean")
    if eligible.empty:
        raise RuntimeError(
            "Nenhum modelo supervisionado encontrado no backtest. "
            f"Esperado um de: {list(PRODUCTION_CANDIDATES)}"
        )

    winner = eligible.iloc[0]["model"]
    src = PRODUCTION_CANDIDATES[winner]
    if not src.exists():
        raise FileNotFoundError(f"Artefato do modelo vencedor não encontrado: {src}")

    dst = PROCESSED_DATA_DIR / "forecast_model.joblib"
    shutil.copyfile(src, dst)

    out = pd.DataFrame(
        [{"selected_model": winner, "criteria": "min_wape_mean_backtest_ml_only", "artifact": str(dst)}]
    )
    out.to_csv(PROCESSED_DATA_DIR / "forecast_model_selection.csv", index=False)

    print(f"Modelo de produção selecionado: {winner}")
    print(f"Artefato de produção: {dst}")


if __name__ == "__main__":
    main()

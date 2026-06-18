from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.models.forecast_features import build_forecast_features
from cosmeticos_ia.models.metrics import mape, wape, temporal_split_by_rows

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _eval_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    pred = pd.Series(model.predict(X_test), index=y_test.index)
    return {
        "model": name,
        "mae": mean_absolute_error(y_test, pred),
        "mape": mape(y_test, pred),
        "wape": wape(y_test, pred),
        "pred": pred,
        "obj": model,
    }


def _eval_naive(name: str, pred: pd.Series, y_test: pd.Series) -> dict:
    """Avalia baseline sem treino (predição pré-calculada)."""
    return {
        "model": name,
        "mae": mean_absolute_error(y_test, pred),
        "mape": mape(y_test, pred),
        "wape": wape(y_test, pred),
        "pred": pred,
        "obj": None,
    }


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    kpis = pd.read_parquet(PROCESSED_DATA_DIR / "kpis_diarios.parquet")
    ds = build_forecast_features(kpis)

    feature_cols = [
        "dia_semana",
        "mes",
        "ano",
        "semana_ano",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "numero_pedidos",
        "clientes_unicos",
        "valor_pedido_fabrica",
        "ticket_medio",
        "fat_lag_1",
        "fat_lag_7",
        "fat_lag_14",
        "fat_lag_21",
        "fat_lag_28",
        "fat_roll_mean_7",
        "fat_roll_mean_14",
        "fat_roll_mean_30",
        "fat_roll_std_7",
        "fat_roll_std_14",
        "fat_roll_std_30",
    ]
    target_col = "faturamento_total"

    train_df, test_df = temporal_split_by_rows(ds, train_ratio=0.8)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    results = []

    # ------------------------------------------------------------------
    # Baselines naives (sem treino — usa features já calculadas)
    # ------------------------------------------------------------------
    if "fat_lag_7" in test_df.columns:
        results.append(_eval_naive("naive_lag7", test_df["fat_lag_7"], y_test))

    if "fat_roll_mean_7" in test_df.columns:
        results.append(_eval_naive("naive_rolling7", test_df["fat_roll_mean_7"], y_test))

    # ------------------------------------------------------------------
    # Modelos supervisionados
    # ------------------------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    results.append(_eval_model("random_forest", rf, X_train, y_train, X_test, y_test))
    rf_model = rf

    xgb_model = None
    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        results.append(_eval_model("xgboost", xgb, X_train, y_train, X_test, y_test))
        xgb_model = xgb

    # Melhor modelo supervisionado (critério: menor WAPE)
    ml_results = [r for r in results if r["obj"] is not None]
    best = sorted(ml_results, key=lambda x: x["wape"])[0]

    # Uplift vs melhor baseline
    baseline_wapes = [r["wape"] for r in results if r["obj"] is None]
    best_baseline_wape = min(baseline_wapes) if baseline_wapes else None

    # ------------------------------------------------------------------
    # Comparação completa
    # ------------------------------------------------------------------
    comp = pd.DataFrame(
        [{k: v for k, v in r.items() if k in ["model", "mae", "mape", "wape"]} for r in results]
    )
    if best_baseline_wape is not None:
        comp["wape_uplift_vs_baseline"] = best_baseline_wape - comp["wape"]
    comp.to_csv(PROCESSED_DATA_DIR / "forecast_model_comparison.csv", index=False)

    pred_out = test_df[["data", target_col]].copy()
    pred_out["pred_faturamento"] = best["pred"].values
    pred_out.to_csv(PROCESSED_DATA_DIR / "forecast_predictions_test.csv", index=False)

    # Salva candidatos para seleção de produção via backtest
    bundle_best = {
        "model": best["obj"],
        "feature_cols": feature_cols,
        "source": "experimental_split",
    }
    joblib.dump(bundle_best, PROCESSED_DATA_DIR / "forecast_model_experimental.joblib")
    joblib.dump(bundle_best, PROCESSED_DATA_DIR / "forecast_model.joblib")

    joblib.dump(
        {"model": rf_model, "feature_cols": feature_cols, "source": "candidate_random_forest"},
        PROCESSED_DATA_DIR / "forecast_model_random_forest.joblib",
    )
    if HAS_XGB and xgb_model is not None:
        joblib.dump(
            {"model": xgb_model, "feature_cols": feature_cols, "source": "candidate_xgboost"},
            PROCESSED_DATA_DIR / "forecast_model_xgboost.joblib",
        )

    metrics_out = pd.DataFrame(
        [
            {
                "model": best["model"],
                "mae": best["mae"],
                "mape": best["mape"],
                "wape": best["wape"],
                "wape_uplift_vs_best_baseline": (
                    best_baseline_wape - best["wape"] if best_baseline_wape is not None else None
                ),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start": train_df["data"].min(),
                "train_end": train_df["data"].max(),
                "test_start": test_df["data"].min(),
                "test_end": test_df["data"].max(),
            }
        ]
    )
    metrics_out.to_csv(PROCESSED_DATA_DIR / "forecast_metrics.csv", index=False)

    print("Treino de previsão concluído.")
    print(comp.to_string(index=False))
    print(
        f"\nModelo vencedor: {best['model']} | WAPE={best['wape']:.4f}"
    )
    if best_baseline_wape is not None:
        print(
            f"Melhor baseline (naive): WAPE={best_baseline_wape:.4f} | "
            f"Uplift: {best_baseline_wape - best['wape']:+.4f}"
        )


if __name__ == "__main__":
    main()

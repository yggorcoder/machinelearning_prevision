from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.models.forecast_features import build_forecast_features

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = y_true.replace(0, pd.NA).abs()
    return float(((y_true - y_pred).abs() / denom).dropna().mean())


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    num = (y_true - y_pred).abs().sum()
    den = y_true.abs().sum()
    return float(num / den) if den != 0 else 0.0


def walk_forward_splits(n_rows: int, initial_train: int, test_size: int, step: int):
    start = initial_train
    while start + test_size <= n_rows:
        yield (0, start, start, start + test_size)
        start += step


def eval_one_model(model_name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    pred = pd.Series(model.predict(X_test), index=y_test.index)
    return {
        "model": model_name,
        "mae": mean_absolute_error(y_test, pred),
        "mape": mape(y_test, pred),
        "wape": wape(y_test, pred),
    }


def main() -> None:
    kpis = pd.read_parquet(PROCESSED_DATA_DIR / "kpis_diarios.parquet")
    ds = build_forecast_features(kpis).sort_values("data").reset_index(drop=True)

    feature_cols = [
        "dia_semana", "mes", "ano", "semana_ano", "is_weekend", "is_month_start", "is_month_end",
        "numero_pedidos", "clientes_unicos", "valor_pedido_fabrica", "ticket_medio",
        "fat_lag_1", "fat_lag_7", "fat_lag_14", "fat_lag_21", "fat_lag_28",
        "fat_roll_mean_7", "fat_roll_mean_14", "fat_roll_mean_30",
        "fat_roll_std_7", "fat_roll_std_14", "fat_roll_std_30",
    ]
    target_col = "faturamento_total"

    n = len(ds)
    initial_train = int(n * 0.60)
    test_size = 30
    step = 30

    rows = []
    fold = 0
    for tr_start, tr_end, te_start, te_end in walk_forward_splits(n, initial_train, test_size, step):
        fold += 1
        train_df = ds.iloc[tr_start:tr_end]
        test_df = ds.iloc[te_start:te_end]

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        rf = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1)
        res_rf = eval_one_model("random_forest", rf, X_train, y_train, X_test, y_test)
        res_rf["fold"] = fold
        rows.append(res_rf)

        if HAS_XGB:
            xgb = XGBRegressor(
                n_estimators=600, max_depth=6, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9,
                objective="reg:squarederror", random_state=42
            )
            res_xgb = eval_one_model("xgboost", xgb, X_train, y_train, X_test, y_test)
            res_xgb["fold"] = fold
            rows.append(res_xgb)

    detail = pd.DataFrame(rows).sort_values(["model", "fold"])
    summary = (
        detail.groupby("model", as_index=False)
        .agg(
            folds=("fold", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            wape_mean=("wape", "mean"),
            wape_std=("wape", "std"),
        )
        .sort_values("wape_mean")
    )

    detail.to_csv(PROCESSED_DATA_DIR / "forecast_backtest_detail.csv", index=False)
    summary.to_csv(PROCESSED_DATA_DIR / "forecast_backtest_summary.csv", index=False)

    print("Backtest concluído.")
    print("\nResumo:")
    print(summary.to_string(index=False))
    print(f"\nDetalhe: {PROCESSED_DATA_DIR / 'forecast_backtest_detail.csv'}")
    print(f"Resumo:  {PROCESSED_DATA_DIR / 'forecast_backtest_summary.csv'}")


if __name__ == "__main__":
    main()

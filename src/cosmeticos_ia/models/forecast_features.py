from __future__ import annotations

import pandas as pd


def build_forecast_features(kpis: pd.DataFrame) -> pd.DataFrame:
    df = kpis.copy().sort_values("data")
    df["data"] = pd.to_datetime(df["data"])

    df["dia_semana"] = df["data"].dt.dayofweek
    df["mes"] = df["data"].dt.month
    df["ano"] = df["data"].dt.year
    df["semana_ano"] = df["data"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dia_semana"] >= 5).astype(int)
    df["is_month_start"] = df["data"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["data"].dt.is_month_end.astype(int)

    for lag in [1, 7, 14, 21, 28]:
        df[f"fat_lag_{lag}"] = df["faturamento_total"].shift(lag)

    for w in [7, 14, 30]:
        df[f"fat_roll_mean_{w}"] = df["faturamento_total"].rolling(w, min_periods=1).mean().shift(1)
        df[f"fat_roll_std_{w}"] = (
            df["faturamento_total"].rolling(w, min_periods=2).std(ddof=0).shift(1).fillna(0.0)
        )

    return df.dropna().reset_index(drop=True)


def build_next_day_feature_row(work: pd.DataFrame, next_date: pd.Timestamp) -> dict:
    row = {
        "dia_semana": next_date.dayofweek,
        "mes": next_date.month,
        "ano": next_date.year,
        "semana_ano": int(next_date.isocalendar().week),
        "is_weekend": int(next_date.dayofweek >= 5),
        "is_month_start": int(next_date.is_month_start),
        "is_month_end": int(next_date.is_month_end),
        "numero_pedidos": float(work["numero_pedidos"].tail(7).mean()),
        "clientes_unicos": float(work["clientes_unicos"].tail(7).mean()),
        "valor_pedido_fabrica": float(work["valor_pedido_fabrica"].tail(7).mean()),
        "ticket_medio": float(work["ticket_medio"].tail(7).mean()),
    }

    last_val = float(work["faturamento_total"].iloc[-1])
    n = len(work)

    for lag in [1, 7, 14, 21, 28]:
        row[f"fat_lag_{lag}"] = float(work["faturamento_total"].iloc[-lag]) if n >= lag else last_val

    for w in [7, 14, 30]:
        tail = work["faturamento_total"].tail(min(w, n))
        row[f"fat_roll_mean_{w}"] = float(tail.mean())
        row[f"fat_roll_std_{w}"] = float(tail.std(ddof=0)) if len(tail) > 1 else 0.0

    return row

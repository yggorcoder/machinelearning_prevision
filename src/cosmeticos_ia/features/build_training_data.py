from __future__ import annotations

import pandas as pd


def build_propensity_dataset(
    df_compras: pd.DataFrame,
    snapshot_freq: str = "W-MON",
    horizon_days: int = 30,
    lookback_days: int = 90,
    short_lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Cria dataset supervisionado para propensão de recompra.

    Cada linha representa (id_cliente, snapshot_date), com features históricas
    até snapshot_date e target indicando compra no horizonte futuro.

    Features geradas
    ----------------
    - recency                  : dias desde a última compra
    - frequency_lookback       : pedidos únicos no lookback
    - monetary_lookback        : valor total no lookback
    - ticket_medio_lookback    : monetary / frequency
    - tendencia_gasto_lookback : monetary_30d / monetary_lookback (aceleração recente)
    """
    df = df_compras.copy()
    df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")
    df = df.dropna(subset=["id_cliente", "id_pedido", "data_compra"]).copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "id_cliente", "snapshot_date", "recency",
                "frequency_lookback", "monetary_lookback",
                "ticket_medio_lookback", "tendencia_gasto_lookback",
                "target_recompra_30d",
            ]
        )

    df["id_cliente"] = df["id_cliente"].astype(str)

    start_date = df["data_compra"].min().normalize()
    end_date = df["data_compra"].max().normalize()

    snapshots = pd.date_range(start=start_date, end=end_date, freq=snapshot_freq)

    rows = []
    for snap_date in snapshots:
        hist_start = snap_date - pd.Timedelta(days=lookback_days)
        short_start = snap_date - pd.Timedelta(days=short_lookback_days)
        future_end = snap_date + pd.Timedelta(days=horizon_days)

        df_hist = df[(df["data_compra"] >= hist_start) & (df["data_compra"] <= snap_date)]
        if df_hist.empty:
            continue

        feats = (
            df_hist.groupby("id_cliente")
            .agg(
                frequency_lookback=("id_pedido", "nunique"),
                monetary_lookback=("valor_total", "sum"),
                last_purchase=("data_compra", "max"),
            )
            .reset_index()
        )
        feats["snapshot_date"] = snap_date
        feats["recency"] = (snap_date - feats["last_purchase"]).dt.days
        feats["ticket_medio_lookback"] = (
            feats["monetary_lookback"] / feats["frequency_lookback"]
        )

        # tendencia: gasto nos últimos short_lookback vs gasto total no lookback
        df_short = df[
            (df["data_compra"] >= short_start) & (df["data_compra"] <= snap_date)
        ]
        monetary_short = (
            df_short.groupby("id_cliente")["valor_total"]
            .sum()
            .rename("monetary_short")
        )
        feats = feats.merge(monetary_short, on="id_cliente", how="left")
        feats["monetary_short"] = feats["monetary_short"].fillna(0.0)
        # ratio: 1.0 = tudo concentrado no período recente; 0.0 = nada recente
        feats["tendencia_gasto_lookback"] = (
            feats["monetary_short"] / feats["monetary_lookback"].replace(0, pd.NA)
        ).fillna(0.0)

        df_future = df[
            (df["data_compra"] > snap_date) & (df["data_compra"] <= future_end)
        ]
        buyers_future = set(df_future["id_cliente"].unique())
        feats["target_recompra_30d"] = feats["id_cliente"].isin(buyers_future).astype(int)

        rows.append(
            feats[
                [
                    "id_cliente",
                    "snapshot_date",
                    "recency",
                    "frequency_lookback",
                    "monetary_lookback",
                    "ticket_medio_lookback",
                    "tendencia_gasto_lookback",
                    "target_recompra_30d",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "id_cliente",
                "snapshot_date",
                "recency",
                "frequency_lookback",
                "monetary_lookback",
                "ticket_medio_lookback",
                "tendencia_gasto_lookback",
                "target_recompra_30d",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["snapshot_date", "id_cliente"]).reset_index(drop=True)
    return out

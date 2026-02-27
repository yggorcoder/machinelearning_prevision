from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.data.loaders import load_clientes, load_compras
from cosmeticos_ia.data.preprocessing import preprocess_clientes, preprocess_compras


def build_scoring_base(df_compras: pd.DataFrame, as_of_date: pd.Timestamp, lookback_days: int = 90) -> pd.DataFrame:
    hist_start = as_of_date - pd.Timedelta(days=lookback_days)
    df_hist = df_compras[(df_compras["data_compra"] >= hist_start) & (df_compras["data_compra"] <= as_of_date)].copy()

    if df_hist.empty:
        return pd.DataFrame(
            columns=[
                "id_cliente", "recency", "frequency_lookback", "monetary_lookback", "ticket_medio_lookback"
            ]
        )

    feats = (
        df_hist.groupby("id_cliente")
        .agg(
            frequency_lookback=("id_pedido", "nunique"),
            monetary_lookback=("valor_total", "sum"),
            last_purchase=("data_compra", "max"),
        )
        .reset_index()
    )
    feats["recency"] = (as_of_date - feats["last_purchase"]).dt.days
    feats["ticket_medio_lookback"] = feats["monetary_lookback"] / feats["frequency_lookback"]
    return feats[["id_cliente", "recency", "frequency_lookback", "monetary_lookback", "ticket_medio_lookback"]]


def main() -> None:
    model_path = PROCESSED_DATA_DIR / "propensity_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = joblib.load(model_path)

    compras = preprocess_compras(load_compras())
    clientes = preprocess_clientes(load_clientes())

    as_of_date = compras["data_compra"].max().normalize()
    scoring = build_scoring_base(compras, as_of_date=as_of_date, lookback_days=90)

    if scoring.empty:
        raise RuntimeError("Base de scoring vazia.")

    feature_cols = ["recency", "frequency_lookback", "monetary_lookback", "ticket_medio_lookback"]
    scoring["score_recompra_30d"] = model.predict_proba(scoring[feature_cols])[:, 1]

    out = scoring.merge(
        clientes[["id_cliente", "nome_cliente"]].drop_duplicates("id_cliente"),
        on="id_cliente",
        how="left",
    ).sort_values("score_recompra_30d", ascending=False)

    out["rank"] = range(1, len(out) + 1)
        # regra simples de prioridade comercial
    out["prioridade"] = pd.cut(
        out["score_recompra_30d"],
        bins=[-1, 0.35, 0.60, 1.0],
        labels=["Baixa", "Media", "Alta"],
    )

    # sugestão de ação para operação
    out["acao_sugerida"] = "Contato padrao"
    out.loc[(out["prioridade"] == "Alta") & (out["recency"] <= 14), "acao_sugerida"] = "Oferta premium"
    out.loc[(out["prioridade"] == "Alta") & (out["recency"] > 14), "acao_sugerida"] = "Reativacao imediata"
    out.loc[(out["prioridade"] == "Media"), "acao_sugerida"] = "Campanha de relacionamento"

    cols = [
        "rank",
        "id_cliente",
        "nome_cliente",
        "score_recompra_30d",
        "prioridade",
        "acao_sugerida",
    ] + feature_cols
    out = out[cols]

    output_path = PROCESSED_DATA_DIR / "propensity_scoring.csv"
    out.to_csv(output_path, index=False)

    # lista operacional da semana (Top 50)
    top50 = out.head(50).copy()
    top50_path = PROCESSED_DATA_DIR / "campanha_top50.csv"
    top50.to_csv(top50_path, index=False)


    print(f"Scoring gerado com {len(out)} clientes.")
    print(f"Arquivo: {output_path}")
    print(f"Top 50 campanha: {top50_path}")
    print("\nTop 10:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

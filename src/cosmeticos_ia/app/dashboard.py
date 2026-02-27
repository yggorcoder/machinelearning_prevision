from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cosmeticos_ia.config import PROCESSED_DATA_DIR

st.set_page_config(page_title="Cosmeticos IA Dashboard", layout="wide")
st.title("Cosmeticos IA - Dashboard Executivo V3")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# Dados
kpi_df = load_parquet(PROCESSED_DATA_DIR / "kpis_diarios.parquet")
prop_metrics = load_csv(PROCESSED_DATA_DIR / "propensity_metrics.csv")
forecast_metrics = load_csv(PROCESSED_DATA_DIR / "forecast_metrics.csv")
campanha_top50 = load_csv(PROCESSED_DATA_DIR / "campanha_top50.csv")
forecast_30d = load_csv(PROCESSED_DATA_DIR / "forecast_future_30d.csv")
forecast_test = load_csv(PROCESSED_DATA_DIR / "forecast_predictions_test.csv")
recall_snapshot = load_csv(PROCESSED_DATA_DIR / "propensity_recall_at_k_by_snapshot.csv")

# Normalização de datas
if not kpi_df.empty and "data" in kpi_df.columns:
    kpi_df["data"] = pd.to_datetime(kpi_df["data"])

if not forecast_30d.empty and "data" in forecast_30d.columns:
    forecast_30d["data"] = pd.to_datetime(forecast_30d["data"])

if not forecast_test.empty and "data" in forecast_test.columns:
    forecast_test["data"] = pd.to_datetime(forecast_test["data"])

# Sidebar
st.sidebar.header("Filtros")
prioridades_disponiveis = []
if not campanha_top50.empty and "prioridade" in campanha_top50.columns:
    prioridades_disponiveis = sorted(campanha_top50["prioridade"].dropna().unique().tolist())

prioridades_sel = st.sidebar.multiselect(
    "Prioridade",
    options=prioridades_disponiveis,
    default=prioridades_disponiveis,
)

if not campanha_top50.empty and prioridades_sel and "prioridade" in campanha_top50.columns:
    campanha_filtrada = campanha_top50[campanha_top50["prioridade"].isin(prioridades_sel)].copy()
else:
    campanha_filtrada = campanha_top50.copy()

# Abas
tab_geral, tab_campanha, tab_forecast = st.tabs(
    ["Visao Geral", "Campanhas", "Forecast"]
)

with tab_geral:
    st.subheader("KPIs Gerais")
    if not kpi_df.empty:
        faturamento_total = float(kpi_df["faturamento_total"].sum())
        ticket_medio = float(kpi_df["ticket_medio"].mean())
        clientes_medios = float(kpi_df["clientes_unicos"].mean())

        c1, c2, c3 = st.columns(3)
        c1.metric("Faturamento total historico", f"R$ {faturamento_total:,.2f}")
        c2.metric("Ticket medio historico", f"R$ {ticket_medio:,.2f}")
        c3.metric("Clientes unicos medio/dia", f"{clientes_medios:,.1f}")

        st.line_chart(kpi_df.set_index("data")["faturamento_total"])
    else:
        st.info("kpis_diarios.parquet nao encontrado.")

    st.subheader("Resumo dos Modelos")
    m1, m2 = st.columns(2)
    with m1:
        st.caption("Propensao")
        if not prop_metrics.empty:
            st.dataframe(prop_metrics, use_container_width=True)
        else:
            st.info("propensity_metrics.csv nao encontrado.")
    with m2:
        st.caption("Forecast")
        if not forecast_metrics.empty:
            st.dataframe(forecast_metrics, use_container_width=True)
        else:
            st.info("forecast_metrics.csv nao encontrado.")

with tab_campanha:
    st.subheader("Campanha Comercial (Top50)")

    if not campanha_filtrada.empty:
        total_clientes = int(len(campanha_filtrada))
        score_medio = float(campanha_filtrada["score_recompra_30d"].mean()) if "score_recompra_30d" in campanha_filtrada.columns else 0.0
        sem_nome_pct = (
            float(campanha_filtrada["nome_cliente"].isna().mean()) * 100
            if "nome_cliente" in campanha_filtrada.columns
            else 0.0
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Clientes na lista", f"{total_clientes}")
        k2.metric("Score medio", f"{score_medio:.4f}")
        k3.metric("% sem nome", f"{sem_nome_pct:.1f}%")

        st.dataframe(campanha_filtrada, use_container_width=True)

        st.download_button(
            label="Baixar campanha filtrada (CSV)",
            data=campanha_filtrada.to_csv(index=False).encode("utf-8"),
            file_name="campanha_filtrada.csv",
            mime="text/csv",
        )
    else:
        st.info("campanha_top50.csv nao encontrado ou filtro sem resultado.")

    st.subheader("Recall@K por Snapshot")
    if not recall_snapshot.empty:
        st.dataframe(recall_snapshot, use_container_width=True)
        if {"k", "recall_at_k_mean_by_snapshot"}.issubset(recall_snapshot.columns):
            st.line_chart(recall_snapshot.set_index("k")["recall_at_k_mean_by_snapshot"])
    else:
        st.info("propensity_recall_at_k_by_snapshot.csv nao encontrado.")

with tab_forecast:
    st.subheader("Previsao Futura (30 dias)")
    if not forecast_30d.empty:
        st.line_chart(forecast_30d.set_index("data")["pred_faturamento"])
        st.dataframe(forecast_30d, use_container_width=True)
        st.download_button(
            label="Baixar previsao futura (CSV)",
            data=forecast_30d.to_csv(index=False).encode("utf-8"),
            file_name="forecast_future_30d.csv",
            mime="text/csv",
        )
    else:
        st.info("forecast_future_30d.csv nao encontrado.")

    st.subheader("Comparacao Real x Previsto (Teste)")
    if not forecast_test.empty and {"faturamento_total", "pred_faturamento", "data"}.issubset(forecast_test.columns):
        comp = forecast_test[["data", "faturamento_total", "pred_faturamento"]].set_index("data")
        st.line_chart(comp)
        st.dataframe(comp.reset_index(), use_container_width=True)
    else:
        st.info("forecast_predictions_test.csv nao encontrado.")




from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cosmeticos_ia.config import PROCESSED_DATA_DIR

CAMPAIGN_LIST_PATH = PROCESSED_DATA_DIR / "campanha_top100.csv"
CAMPAIGN_LIST_FALLBACK = PROCESSED_DATA_DIR / "campanha_top50.csv"

st.set_page_config(page_title="Cosméticos IA", layout="wide", page_icon="📊")
st.title("Cosméticos IA — Dashboard Executivo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def _fmt_pp(value: float, decimals: int = 1) -> str:
    return f"{value:+.{decimals}f} p.p."


def _rename_propensity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename = {
        "model": "Estrategia",
        "pr_auc": "Qualidade do ranking (PR-AUC)",
        "roc_auc": "ROC-AUC",
        "recall_at_50": "Recall Top 50",
        "precision_at_50": "Precisao Top 50",
        "uplift_pr_auc_vs_baseline": "Ganho PR-AUC vs baseline",
        "train_rows": "Linhas treino",
        "test_rows": "Linhas teste",
        "train_start": "Inicio treino",
        "train_end": "Fim treino",
        "test_start": "Inicio teste",
        "test_end": "Fim teste",
    }
    cols = [c for c in rename if c in df.columns]
    out = df[cols].rename(columns=rename)
    out["Estrategia"] = out["Estrategia"].replace(
        {"random_forest": "Modelo IA (Random Forest)", "baseline_recency": "Baseline (recencia)"}
    )
    for col in ("Qualidade do ranking (PR-AUC)", "ROC-AUC", "Recall Top 50", "Precisao Top 50"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.4f}")
    return out


def _rename_forecast_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename = {
        "model": "Modelo",
        "mae": "Erro medio (R$/dia)",
        "mape": "MAPE",
        "wape": "Erro de previsao (WAPE)",
        "wape_uplift_vs_best_baseline": "Ganho WAPE vs baseline",
        "train_rows": "Dias treino",
        "test_rows": "Dias teste",
    }
    cols = [c for c in rename if c in df.columns]
    out = df[cols].rename(columns=rename)
    if "Modelo" in out.columns:
        out["Modelo"] = out["Modelo"].replace({"random_forest": "Random Forest", "xgboost": "XGBoost"})
    if "Erro de previsao (WAPE)" in out.columns:
        out["Erro de previsao (WAPE)"] = out["Erro de previsao (WAPE)"].map(lambda x: _fmt_pct(x))
    if "MAPE" in out.columns:
        out["MAPE"] = out["MAPE"].map(lambda x: _fmt_pct(x))
    if "Erro medio (R$/dia)" in out.columns:
        out["Erro medio (R$/dia)"] = out["Erro medio (R$/dia)"].map(lambda x: f"R$ {x:,.2f}")
    return out


# ---------------------------------------------------------------------------
# Carregar artefatos
# ---------------------------------------------------------------------------

kpi_df = load_parquet(PROCESSED_DATA_DIR / "kpis_diarios.parquet")
prop_metrics = load_csv(PROCESSED_DATA_DIR / "propensity_metrics.csv")
forecast_metrics = load_csv(PROCESSED_DATA_DIR / "forecast_metrics.csv")
forecast_backtest = load_csv(PROCESSED_DATA_DIR / "forecast_backtest_summary.csv")
campanha_top100 = load_csv(CAMPAIGN_LIST_PATH)
if campanha_top100.empty:
    campanha_top100 = load_csv(CAMPAIGN_LIST_FALLBACK)
forecast_30d = load_csv(PROCESSED_DATA_DIR / "forecast_future_30d.csv")
forecast_test = load_csv(PROCESSED_DATA_DIR / "forecast_predictions_test.csv")
recall_snapshot = load_csv(PROCESSED_DATA_DIR / "propensity_recall_at_k_by_snapshot.csv")
uplift_summary = load_csv(PROCESSED_DATA_DIR / "campaign_uplift_summary.csv")
uplift_vs_baseline = load_csv(PROCESSED_DATA_DIR / "campaign_uplift_vs_baseline.csv")
monitoring_report = load_csv(PROCESSED_DATA_DIR / "monitoring_report.csv")
monitoring_summary = load_csv(PROCESSED_DATA_DIR / "monitoring_summary.csv")

for df in (kpi_df, forecast_30d, forecast_test):
    if not df.empty and "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"])

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Filtros")
prioridades_disponiveis = []
if not campanha_top100.empty and "prioridade" in campanha_top100.columns:
    prioridades_disponiveis = sorted(campanha_top100["prioridade"].dropna().unique().tolist())

prioridades_sel = st.sidebar.multiselect(
    "Prioridade da campanha",
    options=prioridades_disponiveis,
    default=prioridades_disponiveis,
)

if not campanha_top100.empty and prioridades_sel and "prioridade" in campanha_top100.columns:
    campanha_filtrada = campanha_top100[campanha_top100["prioridade"].isin(prioridades_sel)].copy()
else:
    campanha_filtrada = campanha_top100.copy()

if not monitoring_summary.empty:
    status = monitoring_summary["overall_status"].iloc[0]
    emoji = {"OK": "🟢", "WARNING": "🟡", "ALERT": "🔴"}.get(status, "⚪")
    st.sidebar.markdown(f"**Monitoramento:** {emoji} {status}")

# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------

tab_geral, tab_campanha, tab_forecast, tab_uplift, tab_monitor = st.tabs(
    ["Visao Geral", "Campanhas", "Forecast", "Uplift", "Monitoramento"]
)

# --- Visao Geral ---
with tab_geral:
    st.subheader("Saude do negocio")
    if not kpi_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Faturamento historico", f"R$ {kpi_df['faturamento_total'].sum():,.2f}")
        c2.metric("Ticket medio", f"R$ {kpi_df['ticket_medio'].mean():,.2f}")
        c3.metric("Clientes unicos / dia", f"{kpi_df['clientes_unicos'].mean():,.1f}")
        st.line_chart(kpi_df.set_index("data")["faturamento_total"])
    else:
        st.info("Execute o pipeline para gerar kpis_diarios.parquet.")

    st.subheader("Confianca dos modelos")

    if not prop_metrics.empty:
        rf = prop_metrics[prop_metrics["model"] == "random_forest"]
        base = prop_metrics[prop_metrics["model"] == "baseline_recency"]
        if not rf.empty and not base.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Propensao — PR-AUC",
                f"{rf['pr_auc'].iloc[0]:.3f}",
                delta=_fmt_pp((rf["pr_auc"].iloc[0] - base["pr_auc"].iloc[0]) * 100, 1).replace(" p.p.", " pp"),
            )
            m2.metric("Precisao Top 50", _fmt_pct(rf["precision_at_50"].iloc[0]))
            m3.metric("Recall Top 50", _fmt_pct(rf["recall_at_50"].iloc[0]))
            if not forecast_metrics.empty:
                m4.metric("Forecast — WAPE", _fmt_pct(forecast_metrics["wape"].iloc[0]))
        st.caption("Propensao — comparacao modelo vs baseline")
        st.dataframe(_rename_propensity_metrics(prop_metrics), use_container_width=True, hide_index=True)
    else:
        st.info("propensity_metrics.csv nao encontrado.")

    if not forecast_metrics.empty:
        st.caption("Forecast — modelo de producao")
        st.dataframe(_rename_forecast_metrics(forecast_metrics), use_container_width=True, hide_index=True)

    if not forecast_backtest.empty:
        st.caption("Forecast — backtest walk-forward (media dos folds)")
        bt = forecast_backtest.copy()
        bt["model"] = bt["model"].replace(
            {
                "random_forest": "Random Forest",
                "naive_lag7": "Baseline lag-7",
                "naive_rolling7": "Baseline media movel 7d",
                "xgboost": "XGBoost",
            }
        )
        show_bt = bt[["model", "folds", "wape_mean", "mae_mean"]].rename(
            columns={
                "model": "Modelo",
                "folds": "Folds",
                "wape_mean": "WAPE medio",
                "mae_mean": "MAE medio (R$)",
            }
        )
        if "WAPE medio" in show_bt.columns:
            show_bt["WAPE medio"] = show_bt["WAPE medio"].map(_fmt_pct)
        if "MAE medio (R$)" in show_bt.columns:
            show_bt["MAE medio (R$)"] = show_bt["MAE medio (R$)"].map(lambda x: f"R$ {x:,.2f}")
        st.dataframe(show_bt, use_container_width=True, hide_index=True)

# --- Campanhas ---
with tab_campanha:
    st.subheader("Lista operacional — Top 100")
    st.caption("Gerada pelo Random Forest de propensao. Use o CSV na acao comercial semanal.")

    if not campanha_filtrada.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("Clientes na lista", len(campanha_filtrada))
        if "score_recompra_30d" in campanha_filtrada.columns:
            k2.metric("Score medio", f"{campanha_filtrada['score_recompra_30d'].mean():.4f}")
        if "nome_cliente" in campanha_filtrada.columns:
            k3.metric("Sem nome cadastrado", f"{campanha_filtrada['nome_cliente'].isna().mean() * 100:.1f}%")

        display_cols = [
            c
            for c in [
                "rank", "id_cliente", "nome_cliente", "score_recompra_30d",
                "prioridade", "acao_sugerida", "recency",
            ]
            if c in campanha_filtrada.columns
        ]
        st.dataframe(campanha_filtrada[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Baixar campanha (CSV)",
            campanha_filtrada.to_csv(index=False).encode("utf-8"),
            "campanha_filtrada.csv",
            "text/csv",
        )
    else:
        st.info("campanha_top100.csv nao encontrado. Rode o pipeline para gerar a lista.")

    st.subheader("Recall@K por snapshot")
    if not recall_snapshot.empty:
        chart_df = recall_snapshot.rename(
            columns={"k": "Top K", "recall_at_k_mean_by_snapshot": "Recall medio"}
        )
        st.line_chart(chart_df.set_index("Top K")["Recall medio"])
        st.dataframe(recall_snapshot, use_container_width=True, hide_index=True)
    else:
        st.info("propensity_recall_at_k_by_snapshot.csv nao encontrado.")

# --- Forecast ---
with tab_forecast:
    st.subheader("Previsao — proximos 30 dias")
    if not forecast_30d.empty:
        total_30d = forecast_30d["pred_faturamento"].sum()
        st.metric("Faturamento previsto (30 dias)", f"R$ {total_30d:,.2f}")
        st.line_chart(forecast_30d.set_index("data")["pred_faturamento"])
        st.dataframe(forecast_30d, use_container_width=True, hide_index=True)
        st.download_button(
            "Baixar previsao (CSV)",
            forecast_30d.to_csv(index=False).encode("utf-8"),
            "forecast_future_30d.csv",
            "text/csv",
        )
    else:
        st.info("forecast_future_30d.csv nao encontrado.")

    st.subheader("Validacao — real vs previsto (periodo de teste)")
    if not forecast_test.empty and {"faturamento_total", "pred_faturamento", "data"}.issubset(forecast_test.columns):
        comp = forecast_test[["data", "faturamento_total", "pred_faturamento"]].copy()
        comp.columns = ["data", "Real", "Previsto"]
        st.line_chart(comp.set_index("data"))
        st.dataframe(comp, use_container_width=True, hide_index=True)
    else:
        st.info("forecast_predictions_test.csv nao encontrado.")

# --- Uplift ---
with tab_uplift:
    st.subheader("Valor do modelo vs regra simples (recencia)")
    st.caption(
        "Mostra o ganho de usar o Random Forest em vez de priorizar apenas "
        "quem comprou mais recentemente."
    )

    if not uplift_vs_baseline.empty:
        u1, u2, u3 = st.columns(3)
        row50 = uplift_vs_baseline[uplift_vs_baseline["k"] == 50]
        if not row50.empty:
            u1.metric("Recall Top 50 — ganho", _fmt_pp(row50["recall_uplift_pp"].iloc[0]))
            u2.metric("Precisao Top 50 — ganho", _fmt_pp(row50["precision_uplift_pp"].iloc[0]))
            u3.metric("Lift Top 50 (modelo)", f"{row50['lift_model'].iloc[0]:.2f}x")

        chart = uplift_vs_baseline.set_index("k")[["lift_model", "lift_baseline"]]
        chart.columns = ["Modelo IA", "Baseline recencia"]
        st.line_chart(chart)
        st.dataframe(uplift_vs_baseline, use_container_width=True, hide_index=True)
    else:
        st.info("campaign_uplift_vs_baseline.csv nao encontrado.")

    if not uplift_summary.empty:
        st.subheader("Detalhe por estrategia e tamanho da lista")
        summary_display = uplift_summary.copy()
        summary_display["strategy"] = summary_display["strategy"].replace(
            {"model": "Modelo IA", "baseline_recency": "Baseline recencia"}
        )
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

# --- Monitoramento ---
with tab_monitor:
    st.subheader("Drift de dados e estabilidade do modelo")

    if not monitoring_summary.empty:
        s = monitoring_summary.iloc[0]
        status = s["overall_status"]
        emoji = {"OK": "🟢", "WARNING": "🟡", "ALERT": "🔴"}.get(status, "⚪")
        st.markdown(f"### Status geral: {emoji} **{status}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Features monitoradas", int(s["n_features"]))
        c2.metric("OK", int(s["n_ok"]))
        c3.metric("Avisos", int(s["n_warning"]))
        c4.metric("Alertas", int(s["n_alert"]))

        if status == "ALERT":
            st.warning("Drift significativo detectado. Considere retreinar os modelos na proxima rotina semanal.")
        elif status == "WARNING":
            st.info("Drift leve detectado. Monitore nas proximas semanas.")
        else:
            st.success("Distribuicao das features estavel em relacao ao periodo de treino.")
    else:
        st.info("monitoring_summary.csv nao encontrado.")

    if not monitoring_report.empty:
        st.subheader("Detalhe por feature")
        report = monitoring_report.copy()
        report["status"] = report["status"].map(
            {"OK": "🟢 OK", "WARNING": "🟡 Aviso", "ALERT": "🔴 Alerta"}
        )
        report = report.rename(
            columns={
                "feature": "Feature",
                "ref_mean": "Media treino",
                "current_mean": "Media recente",
                "psi": "PSI",
                "mean_shift_normalized": "Desvio (sigma)",
                "status": "Status",
            }
        )
        st.dataframe(report, use_container_width=True, hide_index=True)
        st.caption("PSI > 0,2 ou desvio > 30% do desvio-padrao disparam alerta de retreino.")

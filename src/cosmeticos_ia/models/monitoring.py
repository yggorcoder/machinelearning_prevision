"""
Monitoramento pós-deploy: detecção de data drift e degradação de performance.

Compara a distribuição das features na janela recente (últimos N dias)
contra a distribuição no período de treino salvo em disco.

Saída
-----
data/processed/monitoring_report.csv  — indicadores de drift por feature
data/processed/monitoring_summary.csv — status geral (OK / WARNING / ALERT)

Uso
---
    python -m cosmeticos_ia.models.monitoring
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.data.loaders import load_compras
from cosmeticos_ia.data.preprocessing import preprocess_compras
from cosmeticos_ia.features.build_training_data import build_propensity_dataset
from cosmeticos_ia.models.train import temporal_split


# Limites de alerta (diferença de média normalizada)
WARN_THRESHOLD = 0.15   # 15 % de desvio na média normalizada → WARNING
ALERT_THRESHOLD = 0.30  # 30 % → ALERT


def _population_stability_index(p: pd.Series, q: pd.Series, n_bins: int = 10) -> float:
    """
    PSI entre distribuição de referência (p) e atual (q).
    PSI < 0.1: estável; 0.1–0.2: leve mudança; > 0.2: shift significativo.
    """
    combined = pd.concat([p, q]).dropna()
    if combined.empty or combined.std() == 0:
        return 0.0

    bins = pd.cut(combined, bins=n_bins, duplicates="drop")
    categories = bins.cat.categories

    p_cut = pd.cut(p, bins=categories).value_counts(normalize=True).sort_index()
    q_cut = pd.cut(q, bins=categories).value_counts(normalize=True).sort_index()

    p_pct = p_cut.reindex(categories).fillna(1e-4)
    q_pct = q_cut.reindex(categories).fillna(1e-4)

    psi_val = 0.0
    for p_i, q_i in zip(p_pct.values, q_pct.values):
        if p_i > 0 and q_i > 0:
            psi_val += (q_i - p_i) * math.log(q_i / p_i)
    return psi_val


def _mean_shift(ref: pd.Series, cur: pd.Series) -> float:
    """Diferença de médias normalizada pelo desvio-padrão de referência."""
    std = ref.std()
    if std == 0:
        return 0.0
    return float(abs(cur.mean() - ref.mean()) / std)


def run_monitoring(recent_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    compras = preprocess_compras(load_compras())
    ds = build_propensity_dataset(compras)

    train_df, _ = temporal_split(ds, train_ratio=0.8)

    # Janela recente: os últimos `recent_days` de todo o dataset
    max_snap = ds["snapshot_date"].max()
    recent_df = ds[ds["snapshot_date"] >= max_snap - pd.Timedelta(days=recent_days)]

    feature_cols = [c for c in [
        "recency",
        "frequency_lookback",
        "monetary_lookback",
        "ticket_medio_lookback",
        "tendencia_gasto_lookback",
    ] if c in ds.columns]

    rows = []
    for feat in feature_cols:
        ref = train_df[feat].dropna()
        cur = recent_df[feat].dropna()
        if ref.empty or cur.empty:
            continue

        shift = _mean_shift(ref, cur)
        psi = _population_stability_index(ref, cur)

        if psi > 0.2 or shift > ALERT_THRESHOLD:
            status = "ALERT"
        elif psi > 0.1 or shift > WARN_THRESHOLD:
            status = "WARNING"
        else:
            status = "OK"

        rows.append({
            "feature": feat,
            "ref_mean": round(float(ref.mean()), 4),
            "ref_std": round(float(ref.std()), 4),
            "current_mean": round(float(cur.mean()), 4),
            "current_std": round(float(cur.std()), 4),
            "mean_shift_normalized": round(shift, 4),
            "psi": round(psi, 4),
            "status": status,
        })

    report = pd.DataFrame(rows)

    # Faturamento diário (forecast drift)
    kpis_path = PROCESSED_DATA_DIR / "kpis_diarios.parquet"
    if kpis_path.exists():
        kpis = pd.read_parquet(kpis_path)
        kpis["data"] = pd.to_datetime(kpis["data"])
        cutoff = int(len(kpis) * 0.8)
        ref_fat = kpis["faturamento_total"].iloc[:cutoff]
        cur_fat = kpis["faturamento_total"].iloc[cutoff:]
        if not ref_fat.empty and not cur_fat.empty:
            shift_fat = _mean_shift(ref_fat, cur_fat)
            psi_fat = _population_stability_index(ref_fat, cur_fat)
            status_fat = (
                "ALERT" if psi_fat > 0.2 or shift_fat > ALERT_THRESHOLD
                else "WARNING" if psi_fat > 0.1 or shift_fat > WARN_THRESHOLD
                else "OK"
            )
            report = pd.concat([
                report,
                pd.DataFrame([{
                    "feature": "faturamento_total",
                    "ref_mean": round(float(ref_fat.mean()), 4),
                    "ref_std": round(float(ref_fat.std()), 4),
                    "current_mean": round(float(cur_fat.mean()), 4),
                    "current_std": round(float(cur_fat.std()), 4),
                    "mean_shift_normalized": round(shift_fat, 4),
                    "psi": round(psi_fat, 4),
                    "status": status_fat,
                }])
            ], ignore_index=True)

    n_alert = int((report["status"] == "ALERT").sum())
    n_warn = int((report["status"] == "WARNING").sum())
    overall = "ALERT" if n_alert > 0 else "WARNING" if n_warn > 0 else "OK"

    summary = pd.DataFrame([{
        "overall_status": overall,
        "n_features": len(report),
        "n_ok": int((report["status"] == "OK").sum()),
        "n_warning": n_warn,
        "n_alert": n_alert,
        "recent_days": recent_days,
        "reference_rows": len(train_df),
        "current_rows": len(recent_df),
    }])

    return report, summary


def main() -> None:
    report, summary = run_monitoring()

    report_path = PROCESSED_DATA_DIR / "monitoring_report.csv"
    summary_path = PROCESSED_DATA_DIR / "monitoring_summary.csv"
    report.to_csv(report_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Monitoramento concluído.")
    print(f"\nStatus geral: {summary['overall_status'].values[0]}")
    print(report.to_string(index=False))
    print(f"\nRelatório: {report_path}")

    overall = summary["overall_status"].values[0]
    if overall == "ALERT":
        print("\n[ALERTA] Drift significativo detectado — considere retreinar os modelos.")
    elif overall == "WARNING":
        print("\n[AVISO] Drift leve detectado — monitore nas próximas semanas.")


if __name__ == "__main__":
    main()

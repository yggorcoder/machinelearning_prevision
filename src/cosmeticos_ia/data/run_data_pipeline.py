from __future__ import annotations

from pathlib import Path

import pandas as pd

from cosmeticos_ia.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from cosmeticos_ia.data.convert_to_parquet import main as convert_to_parquet_main
from cosmeticos_ia.data.loaders import load_clientes, load_compras, load_pedidos_cd
from cosmeticos_ia.data.preprocessing import (
    preprocess_clientes,
    preprocess_compras,
    preprocess_pedidos,
)
from cosmeticos_ia.data.quality import run_quality_checks, save_quality_report
from cosmeticos_ia.features.build_features import build_daily_kpis, build_rfm


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 0) xlsx (raw) -> parquet — necessário em ambientes limpos (ex.: CI)
    print(f"Convertendo xlsx de {RAW_DATA_DIR} para parquet...")
    convert_to_parquet_main()

    # 1) Load raw->processed base datasets
    df_compras_raw = load_compras()
    df_clientes_raw = load_clientes()
    df_pedidos_raw = load_pedidos_cd()

    # 2) Preprocess
    df_compras = preprocess_compras(df_compras_raw)
    df_clientes = preprocess_clientes(df_clientes_raw)
    df_pedidos = preprocess_pedidos(df_pedidos_raw)

    # 3) Data quality
    quality_report = run_quality_checks(df_compras, df_clientes, df_pedidos)
    report_path = PROCESSED_DATA_DIR / "quality_report.csv"
    save_quality_report(quality_report, report_path)

    # bloqueia pipeline apenas se houver erro crítico
    has_error = (
        (not quality_report.empty)
        and (quality_report["severity"].str.lower() == "error").any()
    )
    if has_error:
        raise RuntimeError(
            f"Pipeline abortado: erros críticos de qualidade detectados. "
            f"Verifique: {report_path}"
        )

    # 4) Features
    rfm = build_rfm(df_compras, df_clientes)
    daily_kpis = build_daily_kpis(df_compras, df_pedidos)

    daily_features = daily_kpis.copy()
    daily_features["dia_semana"] = daily_features.index.dayofweek
    daily_features["mes"] = daily_features.index.month
    daily_features["ano"] = daily_features.index.year
    daily_features["fat_rolling_7"] = (
        daily_features["faturamento_total"].rolling(7, min_periods=1).mean()
    )
    daily_features["fat_rolling_30"] = (
        daily_features["faturamento_total"].rolling(30, min_periods=1).mean()
    )

    # 5) Persist outputs
    df_compras.to_parquet(PROCESSED_DATA_DIR / "compras_clientes_clean.parquet", index=False)
    df_clientes.to_parquet(PROCESSED_DATA_DIR / "clientes_clean.parquet", index=False)
    df_pedidos.to_parquet(PROCESSED_DATA_DIR / "pedidos_cd_clean.parquet", index=False)

    rfm.to_parquet(PROCESSED_DATA_DIR / "rfm_clientes.parquet", index=False)

    daily_kpis_reset = daily_kpis.reset_index()
    daily_kpis_reset.to_parquet(PROCESSED_DATA_DIR / "kpis_diarios.parquet", index=False)

    daily_features_reset = daily_features.reset_index()
    daily_features_reset.to_parquet(
        PROCESSED_DATA_DIR / "kpis_diarios_features.parquet", index=False
    )

    print("Pipeline finalizado com sucesso.")
    print(f"Relatório de qualidade: {report_path}")


if __name__ == "__main__":
    main()

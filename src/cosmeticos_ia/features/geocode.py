from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def _normaliza_endereco(valor: str) -> str:
    """Normaliza o texto do endereço para usar como chave no cache."""
    if pd.isna(valor):
        return ""
    return str(valor).strip().lower()


def geocode_clientes(
    df_clientes: pd.DataFrame,
    endereco_col: str = "endereco",
    cache_path: Optional[Path] = None,
    cidade_padrao: str = "Recife, Pernambuco, Brasil",
) -> pd.DataFrame:
    """
    Recebe df_clientes com uma coluna de endereço e devolve DF com colunas lat/lon.

    Usa cache em parquet para não chamar o serviço de geocodificação toda vez.
    """

    # --- validação básica ---
    if endereco_col not in df_clientes.columns:
        raise ValueError(
            f"Coluna de endereço '{endereco_col}' não encontrada em df_clientes. "
            f"Colunas disponíveis: {list(df_clientes.columns)}"
        )

    df = df_clientes.copy()

    # cria coluna auxiliar normalizada
    df["endereco_norm"] = df[endereco_col].map(_normaliza_endereco)

    # define caminho do cache
    if cache_path is None:
        cache_path = Path("data") / "processed" / "geocoded_clientes.parquet"
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # ==========================
    # 1) Carrega cache se existir
    # ==========================
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
    else:
        cache = pd.DataFrame(columns=["endereco_norm", "lat", "lon"])

    # garante normalização no cache
    if "endereco_norm" not in cache.columns:
        # caso antigo, tenta reconstruir
        if endereco_col in cache.columns:
            cache["endereco_norm"] = cache[endereco_col].map(_normaliza_endereco)
        else:
            cache["endereco_norm"] = ""

    cache["endereco_norm"] = cache["endereco_norm"].map(_normaliza_endereco)

    ja_geocodificados = set(cache["endereco_norm"].dropna())

    # ==========================
    # 2) Endereços novos
    # ==========================
    novos = (
        df[~df["endereco_norm"].isin(ja_geocodificados)]
        [["endereco_norm", endereco_col]]
        .dropna()
        .drop_duplicates("endereco_norm")
    )

    # ==========================
    # 3) Geocodifica apenas novos
    # ==========================
    if not novos.empty:
        # timeout maior (5s) para evitar muitos read timeouts
        geolocator = Nominatim(user_agent="cosmeticos_ia_geocoder", timeout=5)

        # swallow_exceptions=True faz o RateLimiter NÃO estourar erro pro seu código
        geocode = RateLimiter(
            geolocator.geocode,
            min_delay_seconds=1,
            max_retries=2,
            error_wait_seconds=2.0,
            swallow_exceptions=True,
        )

        resultados = []
        for _, row in novos.iterrows():
            end_norm = row["endereco_norm"]
            end_raw = row[endereco_col]

            if not end_norm:
                continue

            query = f"{end_raw}, {cidade_padrao}"

            location = geocode(
                query
            )  # se der erro, volta None por causa do swallow_exceptions

            if location is not None:
                resultados.append(
                    {
                        "endereco_norm": end_norm,
                        "lat": location.latitude,
                        "lon": location.longitude,
                    }
                )

        if resultados:
            novos_cache = pd.DataFrame(resultados)
            cache = pd.concat([cache, novos_cache], ignore_index=True)
            cache = cache.drop_duplicates("endereco_norm")
            cache.to_parquet(cache_path, index=False)

    # ==========================
    # 4) Junta de volta com df_clientes
    # ==========================
    cache_reduzido = cache[["endereco_norm", "lat", "lon"]].drop_duplicates(
        "endereco_norm"
    )

    df = df.merge(cache_reduzido, on="endereco_norm", how="left")

    # remove coluna auxiliar
    df = df.drop(columns=["endereco_norm"])

    return df

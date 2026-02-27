import pandas as pd
from pandas import DataFrame
from cosmeticos_ia.config import (
    COMPRAS_PARQUET,
    CLIENTES_PARQUET,
    PEDIDOS_PARQUET,
)

def load_compras() -> DataFrame:
    return pd.read_parquet(COMPRAS_PARQUET)

def load_clientes() -> DataFrame:
    return pd.read_parquet(CLIENTES_PARQUET)

def load_pedidos_cd() -> DataFrame:
    return pd.read_parquet(PEDIDOS_PARQUET)
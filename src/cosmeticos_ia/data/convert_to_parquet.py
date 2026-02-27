import pandas as pd
from cosmeticos_ia.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    COMPRAS_XLSX,
    CLIENTES_XLSX,
    PEDIDOS_XLSX,
    COMPRAS_PARQUET,
    CLIENTES_PARQUET,
    PEDIDOS_PARQUET,
)

def convert_compras():
    """
    Lê o Excel de compras e salva em Parquet.

    - Linha 0 é lixo → ignoramos
    - Linha 1 contém os nomes das colunas → usamos como header
    - Lemos tudo como string (dtype=str) para evitar problemas de tipo misto
      e arrumamos os tipos depois no preprocessing.
    """
    df = pd.read_excel(COMPRAS_XLSX, dtype=str)

    new_header = df.iloc[1].copy()
    new_header = new_header.fillna("").astype(str).str.strip()

    df = df.iloc[2:].copy()
    df.columns = new_header

    df = df.loc[:, df.columns != ""]

    df.columns = df.columns.str.strip()
   
    df.to_parquet(COMPRAS_PARQUET, index=False)
    print(f"Salvo: {COMPRAS_PARQUET}")

def convert_clientes():
    df = pd.read_excel(CLIENTES_XLSX, dtype=str)

    new_header = df.iloc[1].copy()
    new_header = new_header.fillna("").astype(str).str.strip()

    df = df.iloc[2:].copy()
    df.columns = new_header

    df = df.loc[:, df.columns != ""]

    df.columns = df.columns.str.strip()
    df.to_parquet(CLIENTES_PARQUET, index=False)
    print(f"Salvo: {CLIENTES_PARQUET}")

def convert_pedidos():
    df = pd.read_excel(PEDIDOS_XLSX, dtype=str)
    new_header = df.iloc[1].copy()
    new_header = new_header.fillna("").astype(str).str.strip()

    df = df.iloc[2:].copy()
    df.columns = new_header

    df = df.loc[:, df.columns != ""]

    df.columns = df.columns.str.strip()
    df.to_parquet(PEDIDOS_PARQUET, index=False)
    print(f"Salvo: {PEDIDOS_PARQUET}")

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    convert_compras()
    convert_clientes()
    convert_pedidos()

if __name__ == "__main__":
    main()
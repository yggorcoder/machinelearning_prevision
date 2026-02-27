from pathlib import Path
import os

# BASE-DIR = raiz do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

# Pastas de dados

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = Path(os.getenv("COSMETICOS_RAW_DATA_DIR", str(DATA_DIR / "raw")))
INTERIM_DATA_DIR = Path(os.getenv("COSMETICOS_INTERIM_DATA_DIR", str(DATA_DIR / "interim")))
PROCESSED_DATA_DIR = Path(os.getenv("COSMETICOS_PROCESSED_DATA_DIR", str(DATA_DIR / "processed")))

# Arquivos brutos (.xlsx)
COMPRAS_XLSX = RAW_DATA_DIR / "compras_clientes.xlsx"
CLIENTES_XLSX = RAW_DATA_DIR / "clientes.xlsx"
PEDIDOS_XLSX = RAW_DATA_DIR / "pedidos_cd.xlsx"

# Arquivos processados (.parquet)
COMPRAS_PARQUET = PROCESSED_DATA_DIR / "compras_clientes.parquet"
CLIENTES_PARQUET = PROCESSED_DATA_DIR / "clientes.parquet"
PEDIDOS_PARQUET = PROCESSED_DATA_DIR / "pedidos_cd.parquet"

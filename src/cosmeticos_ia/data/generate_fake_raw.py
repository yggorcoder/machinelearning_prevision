from __future__ import annotations

import argparse
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

from cosmeticos_ia.config import RAW_DATA_DIR


FAKE_RAW_DIR = RAW_DATA_DIR.parent / "raw_fake"
RAW_FILES = ("clientes.xlsx", "compras_clientes.xlsx", "pedidos_cd.xlsx")
HEADER_HINTS = (
    "cod",
    "cód",
    "codigo",
    "nome",
    "data",
    "valor",
    "qtde",
    "quant",
    "pedido",
    "cpf",
    "email",
    "telefone",
    "origem",
    "status",
    "nivel",
    "endereco",
)


def _fake_name(rng: np.random.Generator) -> str:
    first = [
        "Ana",
        "Bruno",
        "Carla",
        "Diego",
        "Elisa",
        "Fabio",
        "Gabi",
        "Helena",
        "Igor",
        "Julia",
    ]
    last = [
        "Silva",
        "Souza",
        "Oliveira",
        "Pereira",
        "Santos",
        "Costa",
        "Lima",
        "Almeida",
        "Gomes",
        "Ribeiro",
    ]
    return f"{rng.choice(first)} {rng.choice(last)}"


def _fake_phone(rng: np.random.Generator) -> str:
    ddd = int(rng.integers(11, 99))
    n1 = int(rng.integers(90000, 99999))
    n2 = int(rng.integers(1000, 9999))
    return f"({ddd}) {n1}-{n2}"


def _fake_cpf(rng: np.random.Generator) -> str:
    n = [int(x) for x in rng.integers(0, 10, size=11)]
    return f"{n[0]}{n[1]}{n[2]}.{n[3]}{n[4]}{n[5]}.{n[6]}{n[7]}{n[8]}-{n[9]}{n[10]}"


def _fake_cep(rng: np.random.Generator) -> str:
    a = int(rng.integers(10000, 99999))
    b = int(rng.integers(100, 999))
    return f"{a}-{b}"


def _fake_date(series: pd.Series, rng: np.random.Generator) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    offsets = rng.integers(-90, 91, size=len(series))
    shifted = parsed + pd.to_timedelta(offsets, unit="D")
    out = series.copy()
    mask = parsed.notna()
    out.loc[mask] = shifted.loc[mask].dt.strftime("%d/%m/%Y")
    return out


def _fake_numeric(series: pd.Series, rng: np.random.Generator) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    num = pd.to_numeric(cleaned, errors="coerce")
    noise = rng.normal(loc=1.0, scale=0.15, size=len(series))
    fake = (num * noise).round(2)
    out = series.copy()
    mask = num.notna()
    out.loc[mask] = fake.loc[mask].map(lambda x: f"{x:.2f}".replace(".", ","))
    return out


def _mask_column(col_name: str, series: pd.Series, rng: np.random.Generator, row_idx: pd.Series) -> pd.Series:
    c = _normalize_text(col_name)
    out = series.copy()

    if any(k in c for k in ("nome",)):
        out.loc[row_idx] = [_fake_name(rng) for _ in range(row_idx.sum())]
        return out
    if any(k in c for k in ("email", "e-mail")):
        out.loc[row_idx] = [f"cliente{i}@example.com" for i in range(1, row_idx.sum() + 1)]
        return out
    if any(k in c for k in ("telefone", "celular", "fone", "whatsapp")):
        out.loc[row_idx] = [_fake_phone(rng) for _ in range(row_idx.sum())]
        return out
    if any(k in c for k in ("cpf", "cnpj", "documento", "rg")):
        out.loc[row_idx] = [_fake_cpf(rng) for _ in range(row_idx.sum())]
        return out
    if any(k in c for k in ("cep",)):
        out.loc[row_idx] = [_fake_cep(rng) for _ in range(row_idx.sum())]
        return out
    if any(k in c for k in ("endereco", "logradouro", "rua", "bairro")):
        streets = ["Rua das Flores", "Av. Central", "Rua do Comercio", "Rua Aurora", "Av. Paulista"]
        nums = rng.integers(10, 999, size=row_idx.sum())
        out.loc[row_idx] = [f"{str(rng.choice(streets))}, {int(n)}" for n in nums]
        return out
    if any(k in c for k in ("cidade",)):
        cities = ["Sao Paulo", "Campinas", "Santos", "Sorocaba", "Ribeirao Preto"]
        out.loc[row_idx] = [str(rng.choice(cities)) for _ in range(row_idx.sum())]
        return out
    if any(k in c for k in ("estado", "uf")):
        ufs = ["SP", "RJ", "MG", "PR", "SC"]
        out.loc[row_idx] = [str(rng.choice(ufs)) for _ in range(row_idx.sum())]
        return out
    if any(k in c for k in ("data", "dt_")):
        out.loc[row_idx] = _fake_date(out.loc[row_idx], rng)
        return out
    if any(k in c for k in ("valor", "preco", "ticket", "qtde", "quantidade", "pontos")):
        out.loc[row_idx] = _fake_numeric(out.loc[row_idx], rng)
        return out
    if any(k in c for k in ("id", "codigo", "cod", "pedido", "op")):
        out.loc[row_idx] = [f"ID{100000 + i}" for i in range(row_idx.sum())]
        return out

    return out


def fake_excel_file(src_path: Path, dst_path: Path, rng: np.random.Generator) -> None:
    df = pd.read_excel(src_path, header=None, dtype=str)

    if df.empty:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(dst_path, header=False, index=False)
        return

    header_row = _detect_header_row(df)
    row_idx = _build_mask_rows(df, header_row)
    col_names = df.iloc[header_row].fillna("").astype(str).tolist()

    out = df.copy()
    for col_pos, col_name in enumerate(col_names):
        out.iloc[:, col_pos] = _mask_column(col_name, out.iloc[:, col_pos], rng, row_idx)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(dst_path, header=False, index=False)


def _detect_header_row(df: pd.DataFrame) -> int:
    scan_rows = min(10, len(df))
    best_idx = 0
    best_score = -1

    for i in range(scan_rows):
        row = df.iloc[i].fillna("").astype(str).map(_normalize_text)
        score = 0
        for cell in row:
            if len(cell) < 2:
                continue
            if any(hint in cell for hint in HEADER_HINTS):
                score += 1
        if score > best_score:
            best_idx = i
            best_score = score

    if best_score < 2 and len(df) > 1:
        return 1
    return best_idx


def _build_mask_rows(df: pd.DataFrame, header_row: int) -> pd.Series:
    mask = pd.Series(df.index > header_row, index=df.index)

    for i in range(header_row):
        row = df.iloc[i].fillna("").astype(str).map(_normalize_text)
        non_empty = int((row != "").sum())
        if non_empty == 0:
            continue
        hints = sum(1 for cell in row if any(hint in cell for hint in HEADER_HINTS))
        if (hints / non_empty) < 0.4:
            mask.loc[i] = True

    return mask


def _normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera planilhas fake a partir dos xlsx originais.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproducibilidade.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Diretorio com os arquivos xlsx reais.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FAKE_RAW_DIR,
        help="Diretorio de saida para os arquivos fake.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for fname in RAW_FILES:
        src = args.input_dir / fname
        if not src.exists():
            print(f"[skip] arquivo nao encontrado: {src}")
            continue
        dst = args.output_dir / fname
        fake_excel_file(src, dst, rng)
        print(f"[ok] gerado: {dst}")


if __name__ == "__main__":
    main()

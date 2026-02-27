from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

@dataclass
class ValidationIssue:
    dataset: str
    rule: str
    severity: str # "error" | "warning"
    details: str

def _null_ratio(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return float(df[col].isna().mean())

def validate_required_columns(
        df: pd.DataFrame, dataset: str, required: Iterable[str]

) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(
            ValidationIssue(
                dataset=dataset,
                rule="required_columns",
                severity="error",
                details=f"Missing required columns: {missing}",
            
            )
        )
    return issues

def validate_duplicates(
        df: pd.DataFrame, dataset: str, key_cols: Iterable[str]

) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    key_cols = list(key_cols)
    if not key_cols or any(c not in df.columns for c in key_cols):
        return issues
    
    dup_count = int(df.duplicated(subset=key_cols).sum())
    if dup_count > 0:
        issues.append(
            ValidationIssue(
                dataset=dataset,
                rule="duplicates",
                severity="warning",
                details=f"{dup_count} duplicated rows on keys {key_cols}",
            )
        )
    return issues


def validate_null_thresholds(
    df: pd.DataFrame, dataset: str, thresholds: dict[str, float]

) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for col, max_ratio in thresholds.items():
        if col not in df.columns:
            continue
        ratio = _null_ratio(df, col)
        if ratio > max_ratio:
            issues.append(
                ValidationIssue(
                    dataset=dataset,
                    rule="null_threshold",
                    severity="warning",
                    details=f"{col}: null_ratio={ratio:.3f} > {max_ratio:.3f}",
                )
            )
    return issues

def validate_date_range(
    df: pd.DataFrame, dataset: str, date_col: str
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if date_col not in df.columns:
        return issues

    series = pd.to_datetime(df[date_col], errors="coerce")
    invalid = int(series.isna().sum())
    if invalid > 0:
        issues.append(
            ValidationIssue(
                dataset=dataset,
                rule="invalid_dates",
                severity="warning",
                details=f"{invalid} invalid/NaT values in {date_col}",
            )
        )
    return issues

def run_quality_checks(
    compras: pd.DataFrame,
    clientes: pd.DataFrame,
    pedidos: pd.DataFrame,
) -> pd.DataFrame:
    issues: list[ValidationIssue] = []

    issues += validate_required_columns(
        compras, "compras", ["id_cliente", "id_pedido", "data_compra", "valor_total"]
    )
    issues += validate_required_columns(
        clientes, "clientes", ["id_cliente", "nome_cliente"]
    )
    issues += validate_required_columns(
        pedidos, "pedidos_cd", ["id_pedido", "data_pedido"]
    )

    issues += validate_duplicates(compras, "compras", ["id_pedido"])
    issues += validate_duplicates(clientes, "clientes", ["id_cliente"])
    issues += validate_duplicates(pedidos, "pedidos_cd", ["id_pedido"])

    issues += validate_null_thresholds(
        compras, "compras", {"id_cliente": 0.01, "valor_total": 0.01}
    )
    issues += validate_null_thresholds(
        clientes, "clientes", {"nome_cliente": 0.20, "data_cadastro": 0.80}
    )
    issues += validate_null_thresholds(
        pedidos, "pedidos_cd", {"data_pedido": 0.05}
    )

    issues += validate_date_range(compras, "compras", "data_compra")
    issues += validate_date_range(clientes, "clientes", "data_cadastro")
    issues += validate_date_range(pedidos, "pedidos_cd", "data_pedido")

    if not issues:
        return pd.DataFrame(columns=["dataset", "rule", "severity", "details"])

    return pd.DataFrame([asdict(i) for i in issues])


def save_quality_report(report_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
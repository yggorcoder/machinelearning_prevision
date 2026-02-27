import pandas as pd

from cosmeticos_ia.data.quality import run_quality_checks


def test_run_quality_checks_no_errors_for_valid_minimal_data():
    compras = pd.DataFrame(
        {
            "id_cliente": ["1", "2"],
            "id_pedido": ["10", "11"],
            "data_compra": pd.to_datetime(["2025-01-10", "2025-01-11"]),
            "valor_total": [100.0, 200.0],
        }
    )

    clientes = pd.DataFrame(
        {
            "id_cliente": ["1", "2"],
            "nome_cliente": ["Ana", "Bia"],
            "data_cadastro": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        }
    )

    pedidos = pd.DataFrame(
        {
            "id_pedido": ["500", "501"],
            "data_pedido": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        }
    )

    report = run_quality_checks(compras, clientes, pedidos)
    assert (report["severity"] == "error").sum() == 0


def test_run_quality_checks_detects_missing_required_columns():
    compras = pd.DataFrame({"id_cliente": ["1"]})  # faltando colunas obrigatórias
    clientes = pd.DataFrame({"id_cliente": ["1"]})  # faltando nome_cliente
    pedidos = pd.DataFrame({"id_pedido": ["1"]})  # faltando data_pedido

    report = run_quality_checks(compras, clientes, pedidos)

    assert not report.empty
    assert "required_columns" in set(report["rule"])
    assert (report["severity"] == "error").sum() >= 1

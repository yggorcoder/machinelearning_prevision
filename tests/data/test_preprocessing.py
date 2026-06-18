import pandas as pd

from cosmeticos_ia.data.preprocessing import preprocess_compras


def test_preprocess_compras_parses_brazilian_money() -> None:
    df = pd.DataFrame(
        {
            "Cód. Cliente": ["100"],
            "OP": ["OP1"],
            "Data": ["15/03/2024"],
            "Valor Total": ["1.234,56"],
            "Valor Desconto": ["10,50"],
            "Qtde Itens": ["3"],
        }
    )
    out = preprocess_compras(df)
    assert out["valor_total"].iloc[0] == 1234.56
    assert out["valor_desconto"].iloc[0] == 10.5

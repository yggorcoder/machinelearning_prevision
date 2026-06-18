"""Testes para build_propensity_dataset — invariantes críticos."""
import numpy as np
import pandas as pd
import pytest

from cosmeticos_ia.features.build_training_data import build_propensity_dataset


def _make_compras(n_clients: int = 10, n_orders: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    return pd.DataFrame({
        "id_cliente": [f"C{rng.integers(1, n_clients + 1):02d}" for _ in range(n_orders)],
        "id_pedido": [f"P{i:04d}" for i in range(n_orders)],
        "data_compra": rng.choice(dates, size=n_orders),
        "valor_total": rng.uniform(50, 500, size=n_orders),
    })


@pytest.fixture
def compras_df():
    return _make_compras()


def test_output_columns(compras_df):
    ds = build_propensity_dataset(compras_df)
    expected = {
        "id_cliente", "snapshot_date", "recency",
        "frequency_lookback", "monetary_lookback",
        "ticket_medio_lookback", "tendencia_gasto_lookback",
        "target_recompra_30d",
    }
    assert expected.issubset(set(ds.columns))


def test_target_is_binary(compras_df):
    ds = build_propensity_dataset(compras_df)
    assert set(ds["target_recompra_30d"].unique()).issubset({0, 1})


def test_recency_non_negative(compras_df):
    ds = build_propensity_dataset(compras_df)
    assert (ds["recency"] >= 0).all()


def test_no_future_leakage(compras_df):
    """Nenhum dado após snapshot_date deve aparecer nas features."""
    ds = build_propensity_dataset(compras_df)
    # frequency e monetary são baseados no lookback; não podem ser negativos
    assert (ds["frequency_lookback"] >= 1).all()
    assert (ds["monetary_lookback"] >= 0).all()


def test_tendencia_range(compras_df):
    ds = build_propensity_dataset(compras_df)
    assert (ds["tendencia_gasto_lookback"] >= 0.0).all()
    assert (ds["tendencia_gasto_lookback"] <= 1.0).all()


def test_temporal_order(compras_df):
    ds = build_propensity_dataset(compras_df)
    sorted_snaps = ds.sort_values("snapshot_date")["snapshot_date"]
    assert list(sorted_snaps) == sorted(ds["snapshot_date"].tolist())


def test_empty_input_returns_empty_df():
    empty = pd.DataFrame(columns=["id_cliente", "id_pedido", "data_compra", "valor_total"])
    ds = build_propensity_dataset(empty)
    assert ds.empty
    assert "target_recompra_30d" in ds.columns


def test_no_nulls_in_key_columns(compras_df):
    ds = build_propensity_dataset(compras_df)
    for col in ["recency", "frequency_lookback", "monetary_lookback", "target_recompra_30d"]:
        assert ds[col].isna().sum() == 0, f"NaN encontrado em {col}"

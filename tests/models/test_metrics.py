"""Testes unitários para o módulo de métricas compartilhadas."""
import pandas as pd
import pytest

from cosmeticos_ia.models.metrics import (
    mape,
    wape,
    pr_auc,
    roc_auc,
    recall_at_k,
    precision_at_k,
    temporal_split_by_date,
    temporal_split_by_rows,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def test_wape_perfect_prediction():
    y = pd.Series([100.0, 200.0, 300.0])
    assert wape(y, y) == pytest.approx(0.0)


def test_wape_known_value():
    y_true = pd.Series([100.0, 100.0])
    y_pred = pd.Series([50.0, 100.0])
    # |50| + |0| = 50; sum(|y|) = 200 → 0.25
    assert wape(y_true, y_pred) == pytest.approx(0.25)


def test_wape_all_zeros_denominator():
    y = pd.Series([0.0, 0.0])
    assert wape(y, y) == pytest.approx(0.0)


def test_mape_ignores_zeros():
    y_true = pd.Series([0.0, 100.0])
    y_pred = pd.Series([10.0, 110.0])
    # só o segundo entra: |10|/100 = 0.10
    assert mape(y_true, y_pred) == pytest.approx(0.10)


def test_mape_perfect():
    y = pd.Series([10.0, 20.0, 30.0])
    assert mape(y, y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def _binary_data():
    y_true = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])
    y_score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    return y_true, y_score


def test_pr_auc_range():
    y, s = _binary_data()
    score = pr_auc(y, s)
    assert 0.0 <= score <= 1.0


def test_roc_auc_range():
    y, s = _binary_data()
    score = roc_auc(y, s)
    assert 0.0 <= score <= 1.0


def test_recall_at_k_perfect():
    # Todos os positivos no topo
    y = pd.Series([1, 1, 1, 0, 0])
    s = pd.Series([0.9, 0.8, 0.7, 0.3, 0.1])
    assert recall_at_k(y, s, k=3) == pytest.approx(1.0)


def test_recall_at_k_zero():
    # Nenhum positivo no top-2
    y = pd.Series([0, 0, 1, 1, 1])
    s = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
    assert recall_at_k(y, s, k=2) == pytest.approx(0.0)


def test_recall_at_k_no_positives():
    y = pd.Series([0, 0, 0])
    s = pd.Series([0.9, 0.5, 0.1])
    assert recall_at_k(y, s, k=2) == pytest.approx(0.0)


def test_precision_at_k_perfect():
    y = pd.Series([1, 1, 0, 0])
    s = pd.Series([0.9, 0.8, 0.3, 0.1])
    assert precision_at_k(y, s, k=2) == pytest.approx(1.0)


def test_precision_at_k_partial():
    y = pd.Series([1, 0, 1, 0])
    s = pd.Series([0.9, 0.8, 0.3, 0.1])
    # top-2: scores 0.9 (y=1) e 0.8 (y=0) → 1 acerto em 2 → 0.5
    assert precision_at_k(y, s, k=2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Temporal splits
# ---------------------------------------------------------------------------

def _snapshot_df():
    dates = pd.date_range("2024-01-07", periods=10, freq="W")
    rows = []
    for d in dates:
        for _ in range(3):
            rows.append({"snapshot_date": d, "x": 1})
    return pd.DataFrame(rows)


def test_temporal_split_by_date_no_leakage():
    df = _snapshot_df()
    train, test = temporal_split_by_date(df, "snapshot_date", train_ratio=0.8)
    assert train["snapshot_date"].max() < test["snapshot_date"].min()


def test_temporal_split_by_date_sizes():
    df = _snapshot_df()
    train, test = temporal_split_by_date(df, "snapshot_date", train_ratio=0.8)
    assert len(train) + len(test) == len(df)
    assert len(train) > 0
    assert len(test) > 0


def test_temporal_split_by_rows_no_overlap():
    df = pd.DataFrame({"v": range(100)})
    train, test = temporal_split_by_rows(df, train_ratio=0.8)
    assert len(train) == 80
    assert len(test) == 20
    assert train.index.max() < test.index.min()

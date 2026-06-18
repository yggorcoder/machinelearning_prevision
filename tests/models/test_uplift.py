"""Testes para avaliação de uplift de campanha."""
import pandas as pd
import pytest

from cosmeticos_ia.models.metrics import recall_at_k, precision_at_k
from cosmeticos_ia.models.evaluate_campaign_uplift import _snapshot_metrics


def _make_snapshot(n: int = 20, positive_rate: float = 0.3, seed: int = 0):
    rng = __import__("random")
    rng.seed(seed)
    y = [1 if rng.random() < positive_rate else 0 for _ in range(n)]
    score = sorted([rng.random() for _ in range(n)], reverse=True)
    return pd.DataFrame({"target_recompra_30d": y, "score_model": score})


def test_snapshot_metrics_keys():
    df = _make_snapshot()
    result = _snapshot_metrics(df, "score_model", k=5)
    assert set(result.keys()) >= {"recall_at_k", "precision_at_k", "lift_at_k", "base_rate"}


def test_lift_at_k_above_one_for_perfect_model():
    """Modelo perfeito deve ter lift >= 1."""
    n = 10
    y = [1] * 3 + [0] * 7
    score = [0.9, 0.8, 0.7] + [0.3] * 7  # top-3 são os positivos
    df = pd.DataFrame({"target_recompra_30d": y, "score_model": score})
    result = _snapshot_metrics(df, "score_model", k=3)
    assert result["lift_at_k"] >= 1.0


def test_recall_at_k_increases_with_k():
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
    score = pd.Series([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55])
    r2 = recall_at_k(y, score, k=2)
    r4 = recall_at_k(y, score, k=4)
    r8 = recall_at_k(y, score, k=8)
    assert r2 <= r4 <= r8


def test_precision_at_k_between_zero_and_one():
    y = pd.Series([1, 0, 0, 1])
    score = pd.Series([0.9, 0.8, 0.7, 0.6])
    p = precision_at_k(y, score, k=2)
    assert 0.0 <= p <= 1.0


def test_recall_equals_one_when_k_ge_all_positives():
    y = pd.Series([1, 1, 0, 0, 0])
    score = pd.Series([0.9, 0.8, 0.3, 0.2, 0.1])
    assert recall_at_k(y, score, k=5) == pytest.approx(1.0)

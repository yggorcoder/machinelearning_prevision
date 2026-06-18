"""Shared evaluation metrics used across propensity and forecast models."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error, ignoring zero-valued actuals."""
    denom = y_true.replace(0, pd.NA).abs()
    return float(((y_true - y_pred).abs() / denom).dropna().mean())


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted Absolute Percentage Error (sum-normalised MAE)."""
    num = (y_true - y_pred).abs().sum()
    den = y_true.abs().sum()
    return float(num / den) if den != 0 else 0.0


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def pr_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    return float(roc_auc_score(y_true, y_score))


def recall_at_k(y_true: pd.Series, y_score: pd.Series, k: int) -> float:
    """Recall among the top-K scored items."""
    df = pd.DataFrame({"y": y_true.values, "score": y_score.values})
    df = df.sort_values("score", ascending=False)
    positives_total = int(df["y"].sum())
    if positives_total == 0:
        return 0.0
    return float(df.head(k)["y"].sum() / positives_total)


def precision_at_k(y_true: pd.Series, y_score: pd.Series, k: int) -> float:
    """Precision among the top-K scored items."""
    df = pd.DataFrame({"y": y_true.values, "score": y_score.values})
    df = df.sort_values("score", ascending=False)
    return float(df.head(k)["y"].mean()) if k > 0 else 0.0


# ---------------------------------------------------------------------------
# Temporal split (shared across propensity and forecast)
# ---------------------------------------------------------------------------

def temporal_split_by_date(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by unique dates, preserving temporal order."""
    dates = sorted(df[date_col].unique())
    cutoff_idx = max(1, int(len(dates) * train_ratio))
    cutoff = dates[cutoff_idx - 1]
    return df[df[date_col] <= cutoff].copy(), df[df[date_col] > cutoff].copy()


def temporal_split_by_rows(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by row position, preserving temporal order."""
    cutoff = max(1, int(len(df) * train_ratio))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()

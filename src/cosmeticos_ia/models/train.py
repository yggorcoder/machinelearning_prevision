from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.data.loaders import load_compras
from cosmeticos_ia.data.preprocessing import preprocess_compras
from cosmeticos_ia.features.build_training_data import build_propensity_dataset
from cosmeticos_ia.models.metrics import (
    pr_auc,
    roc_auc,
    recall_at_k,
    precision_at_k,
    temporal_split_by_date,
)


def temporal_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper — split by snapshot_date."""
    return temporal_split_by_date(df, "snapshot_date", train_ratio)


def _score_naive_recency(df: pd.DataFrame) -> pd.Series:
    """Baseline: quanto menor a recency (mais recente), maior o score."""
    max_recency = df["recency"].max()
    return (max_recency - df["recency"]).astype(float)


def _eval_row(
    name: str,
    y_true: pd.Series,
    y_score: pd.Series,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    row: dict = {
        "model": name,
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": roc_auc(y_true, y_score),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": train_df["snapshot_date"].min(),
        "train_end": train_df["snapshot_date"].max(),
        "test_start": test_df["snapshot_date"].min(),
        "test_end": test_df["snapshot_date"].max(),
    }
    for k in [10, 30, 50]:
        row[f"recall_at_{k}"] = recall_at_k(y_true, y_score, k)
        row[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
    return row


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    compras = preprocess_compras(load_compras())
    ds = build_propensity_dataset(compras)

    if ds.empty:
        raise RuntimeError("Dataset de treinamento vazio.")

    train_df, test_df = temporal_split(ds, train_ratio=0.8)
    if test_df.empty:
        raise RuntimeError("Conjunto de teste vazio após split temporal.")

    feature_cols = [
        "recency",
        "frequency_lookback",
        "monetary_lookback",
        "ticket_medio_lookback",
        "tendencia_gasto_lookback",
    ]
    # fallback: se feature nova ainda não existir no dataset
    feature_cols = [c for c in feature_cols if c in ds.columns]
    target_col = "target_recompra_30d"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # ------------------------------------------------------------------
    # Baseline: naive recency ranking
    # ------------------------------------------------------------------
    naive_score_test = _score_naive_recency(test_df)
    naive_row = _eval_row("baseline_recency", y_test, naive_score_test, train_df, test_df)

    # ------------------------------------------------------------------
    # Modelo principal: RandomForest
    # ------------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    proba_test = pd.Series(model.predict_proba(X_test)[:, 1], index=test_df.index)
    model_row = _eval_row("random_forest", y_test, proba_test, train_df, test_df)

    # ------------------------------------------------------------------
    # Comparação: modelo vs baseline
    # ------------------------------------------------------------------
    metrics_all = pd.DataFrame([naive_row, model_row])
    metrics_all["uplift_pr_auc_vs_baseline"] = (
        metrics_all["pr_auc"] - metrics_all.loc[metrics_all["model"] == "baseline_recency", "pr_auc"].values[0]
    )
    # Linha de produção = modelo
    prod_row = metrics_all[metrics_all["model"] == "random_forest"].iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------
    joblib.dump(model, PROCESSED_DATA_DIR / "propensity_model.joblib")
    ds.to_parquet(PROCESSED_DATA_DIR / "propensity_dataset.parquet", index=False)
    metrics_all.to_csv(PROCESSED_DATA_DIR / "propensity_metrics.csv", index=False)

    print("Treino finalizado.")
    print(metrics_all[["model", "pr_auc", "roc_auc", "recall_at_50", "precision_at_50"]].to_string(index=False))
    print(f"\nUplift PR-AUC vs baseline: {model_row['pr_auc'] - naive_row['pr_auc']:+.4f}")
    print(f"Modelo salvo em: {PROCESSED_DATA_DIR / 'propensity_model.joblib'}")
    print(f"Métricas salvas em: {PROCESSED_DATA_DIR / 'propensity_metrics.csv'}")


if __name__ == "__main__":
    main()

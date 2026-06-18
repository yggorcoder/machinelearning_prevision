"""
Avalia o uplift da campanha Top-K comparando o modelo de propensão
contra o baseline naive (ranking por recency).

Por snapshot de teste, computa para cada K:
  - Recall@K: fração dos compradores reais capturados pelo Top-K
  - Precision@K: taxa de acerto dentro do Top-K
  - Lift@K: Precision@K do modelo / taxa de compra global

Saídas
------
data/processed/campaign_uplift_summary.csv   — médias por (estratégia, K)
data/processed/campaign_uplift_detail.csv    — por (estratégia, K, snapshot)
"""
from __future__ import annotations

import joblib
import pandas as pd

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.data.loaders import load_compras
from cosmeticos_ia.data.preprocessing import preprocess_compras
from cosmeticos_ia.features.build_training_data import build_propensity_dataset
from cosmeticos_ia.models.metrics import temporal_split_by_date, recall_at_k, precision_at_k
from cosmeticos_ia.models.train import temporal_split


K_VALUES = [10, 30, 50]


def _snapshot_metrics(df_snap: pd.DataFrame, score_col: str, k: int) -> dict:
    df = df_snap.sort_values(score_col, ascending=False)
    y = df["target_recompra_30d"]
    score = df[score_col]
    positives = int(y.sum())
    n = len(y)
    base_rate = positives / n if n > 0 else 0.0

    prec = precision_at_k(y, score, k)
    rec = recall_at_k(y, score, k)
    lift = prec / base_rate if base_rate > 0 else 0.0

    return {
        "recall_at_k": rec,
        "precision_at_k": prec,
        "lift_at_k": lift,
        "base_rate": base_rate,
        "positives": positives,
        "n": n,
    }


def main() -> None:
    model_path = PROCESSED_DATA_DIR / "propensity_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = joblib.load(model_path)
    compras = preprocess_compras(load_compras())
    ds = build_propensity_dataset(compras)

    _, test_df = temporal_split(ds, train_ratio=0.8)

    feature_cols = [c for c in [
        "recency",
        "frequency_lookback",
        "monetary_lookback",
        "ticket_medio_lookback",
        "tendencia_gasto_lookback",
    ] if c in test_df.columns]

    X_test = test_df[feature_cols]

    # Scores do modelo
    test_df = test_df.copy()
    test_df["score_model"] = model.predict_proba(X_test)[:, 1]

    # Score do baseline: maior score = menor recency
    max_rec = test_df["recency"].max()
    test_df["score_baseline"] = max_rec - test_df["recency"].astype(float)

    rows = []
    for snap_date, grp in test_df.groupby("snapshot_date"):
        for k in K_VALUES:
            for strategy, score_col in [("model", "score_model"), ("baseline_recency", "score_baseline")]:
                m = _snapshot_metrics(grp, score_col, k)
                rows.append({
                    "snapshot_date": snap_date,
                    "strategy": strategy,
                    "k": k,
                    **m,
                })

    detail = pd.DataFrame(rows)

    summary = (
        detail.groupby(["strategy", "k"], as_index=False)
        .agg(
            snapshots=("snapshot_date", "count"),
            recall_mean=("recall_at_k", "mean"),
            recall_std=("recall_at_k", "std"),
            precision_mean=("precision_at_k", "mean"),
            lift_mean=("lift_at_k", "mean"),
            base_rate_mean=("base_rate", "mean"),
        )
        .sort_values(["k", "strategy"])
    )

    # Uplift do modelo vs baseline por K
    uplift_rows = []
    for k in K_VALUES:
        model_row = summary[(summary["strategy"] == "model") & (summary["k"] == k)]
        base_row = summary[(summary["strategy"] == "baseline_recency") & (summary["k"] == k)]
        if model_row.empty or base_row.empty:
            continue
        uplift_rows.append({
            "k": k,
            "recall_uplift_pp": (
                model_row["recall_mean"].values[0] - base_row["recall_mean"].values[0]
            ) * 100,
            "precision_uplift_pp": (
                model_row["precision_mean"].values[0] - base_row["precision_mean"].values[0]
            ) * 100,
            "lift_model": model_row["lift_mean"].values[0],
            "lift_baseline": base_row["lift_mean"].values[0],
        })
    uplift_summary = pd.DataFrame(uplift_rows)

    detail.to_csv(PROCESSED_DATA_DIR / "campaign_uplift_detail.csv", index=False)
    summary.to_csv(PROCESSED_DATA_DIR / "campaign_uplift_summary.csv", index=False)
    uplift_summary.to_csv(PROCESSED_DATA_DIR / "campaign_uplift_vs_baseline.csv", index=False)

    print("Uplift da campanha calculado.")
    print("\nSuário por estratégia e K:")
    print(summary.to_string(index=False))
    if not uplift_summary.empty:
        print("\nUplift modelo vs baseline (em p.p.):")
        print(uplift_summary.to_string(index=False))


if __name__ == "__main__":
    main()

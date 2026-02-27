from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.data.loaders import load_compras
from cosmeticos_ia.data.preprocessing import preprocess_compras
from cosmeticos_ia.features.build_training_data import build_propensity_dataset


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["snapshot_date"].unique())
    cutoff_idx = max(1, int(len(dates) * train_ratio))
    cutoff_date = dates[cutoff_idx - 1]

    train_df = df[df["snapshot_date"] <= cutoff_date].copy()
    test_df = df[df["snapshot_date"] > cutoff_date].copy()

    return train_df, test_df


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Build training dataset
    compras = preprocess_compras(load_compras())
    ds = build_propensity_dataset(compras)

    if ds.empty:
        raise RuntimeError("Dataset de treinamento vazio.")

    # 2) Split temporal
    train_df, test_df = temporal_split(ds, train_ratio=0.8)
    if test_df.empty:
        raise RuntimeError("Conjunto de teste vazio após split temporal.")

    feature_cols = [
        "recency",
        "frequency_lookback",
        "monetary_lookback",
        "ticket_medio_lookback",
    ]
    target_col = "target_recompra_30d"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 3) Train baseline model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    # 4) Evaluate
    proba_test = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba_test)
    roc_auc = roc_auc_score(y_test, proba_test)

    metrics = pd.DataFrame(
        [
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start": train_df["snapshot_date"].min(),
                "train_end": train_df["snapshot_date"].max(),
                "test_start": test_df["snapshot_date"].min(),
                "test_end": test_df["snapshot_date"].max(),
            }
        ]
    )

    # 5) Persist artifacts
    joblib.dump(model, PROCESSED_DATA_DIR / "propensity_model.joblib")
    ds.to_parquet(PROCESSED_DATA_DIR / "propensity_dataset.parquet", index=False)
    metrics.to_csv(PROCESSED_DATA_DIR / "propensity_metrics.csv", index=False)

    print("Treino finalizado.")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Métricas salvas em: {PROCESSED_DATA_DIR / 'propensity_metrics.csv'}")
    print(f"Modelo salvo em: {PROCESSED_DATA_DIR / 'propensity_model.joblib'}")


if __name__ == "__main__":
    main()

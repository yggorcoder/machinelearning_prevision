from __future__ import annotations

import joblib
import pandas as pd

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.data.loaders import load_compras
from cosmeticos_ia.data.preprocessing import preprocess_compras
from cosmeticos_ia.features.build_training_data import build_propensity_dataset
from cosmeticos_ia.models.metrics import recall_at_k
from cosmeticos_ia.models.train import temporal_split


def recall_at_k_by_snapshot(
    test_df: pd.DataFrame, score: pd.Series, k: int
) -> float:
    aux = test_df[["snapshot_date", "target_recompra_30d"]].copy()
    aux["score"] = score.values

    recalls = [
        recall_at_k(grp["target_recompra_30d"], grp["score"], k=k)
        for _, grp in aux.groupby("snapshot_date")
    ]
    return float(pd.Series(recalls).mean()) if recalls else 0.0


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
    score = pd.Series(model.predict_proba(X_test)[:, 1], index=test_df.index)

    k_values = [10, 30, 50, 100]
    rows = [
        {
            "k": k,
            "recall_at_k_mean_by_snapshot": recall_at_k_by_snapshot(test_df, score, k=k),
        }
        for k in k_values
    ]

    out = pd.DataFrame(rows)
    out_path = PROCESSED_DATA_DIR / "propensity_recall_at_k_by_snapshot.csv"
    out.to_csv(out_path, index=False)

    print("Avaliação por snapshot concluída.")
    print(out.to_string(index=False))
    print(f"\nArquivo: {out_path}")


if __name__ == "__main__":
    main()

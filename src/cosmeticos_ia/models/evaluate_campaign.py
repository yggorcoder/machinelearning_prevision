from __future__ import annotations

import joblib
import pandas as pd

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.data.loaders import load_compras
from cosmeticos_ia.data.preprocessing import preprocess_compras
from cosmeticos_ia.features.build_training_data import build_propensity_dataset
from cosmeticos_ia.models.train import temporal_split


def recall_at_k_single(df: pd.DataFrame, k: int) -> float:
    df = df.sort_values("score", ascending=False)
    topk = df.head(k)
    positives_total = int(df["y"].sum())
    if positives_total == 0:
        return 0.0
    return float(topk["y"].sum() / positives_total)


def recall_at_k_by_snapshot(test_df: pd.DataFrame, score: pd.Series, k: int) -> float:
    aux = test_df[["snapshot_date", "target_recompra_30d"]].copy()
    aux["score"] = score.values
    aux = aux.rename(columns={"target_recompra_30d": "y"})

    recalls = []
    for _, grp in aux.groupby("snapshot_date"):
        recalls.append(recall_at_k_single(grp, k=k))

    if not recalls:
        return 0.0
    return float(pd.Series(recalls).mean())


def main() -> None:
    model_path = PROCESSED_DATA_DIR / "propensity_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = joblib.load(model_path)

    compras = preprocess_compras(load_compras())
    ds = build_propensity_dataset(compras)
    _, test_df = temporal_split(ds, train_ratio=0.8)

    feature_cols = [
        "recency",
        "frequency_lookback",
        "monetary_lookback",
        "ticket_medio_lookback",
    ]
    X_test = test_df[feature_cols]
    score = pd.Series(model.predict_proba(X_test)[:, 1], index=test_df.index)

    k_values = [10, 30, 50, 100]
    rows = []
    for k in k_values:
        rows.append(
            {
                "k": k,
                "recall_at_k_mean_by_snapshot": recall_at_k_by_snapshot(test_df, score, k=k),
            }
        )

    out = pd.DataFrame(rows)
    out_path = PROCESSED_DATA_DIR / "propensity_recall_at_k_by_snapshot.csv"
    out.to_csv(out_path, index=False)

    print("Avaliação por snapshot concluída.")
    print(out.to_string(index=False))
    print(f"\nArquivo: {out_path}")


if __name__ == "__main__":
    main()


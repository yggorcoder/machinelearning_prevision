from __future__ import annotations

import joblib
import pandas as pd

from cosmeticos_ia.config import PROCESSED_DATA_DIR
from cosmeticos_ia.models.forecast_features import build_next_day_feature_row


def main(horizon_days: int = 30) -> None:
    bundle = joblib.load(PROCESSED_DATA_DIR / "forecast_model.joblib")
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    work = pd.read_parquet(PROCESSED_DATA_DIR / "kpis_diarios.parquet").sort_values("data").copy()
    work["data"] = pd.to_datetime(work["data"])

    future_rows = []
    for _ in range(horizon_days):
        next_date = work["data"].max() + pd.Timedelta(days=1)
        row = build_next_day_feature_row(work, next_date)

        x = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(x)[0])

        future_rows.append({"data": next_date, "pred_faturamento": pred})

        work = pd.concat(
            [work, pd.DataFrame([{
                "data": next_date,
                "faturamento_total": pred,
                "numero_pedidos": row["numero_pedidos"],
                "clientes_unicos": row["clientes_unicos"],
                "ticket_medio": row["ticket_medio"],
                "valor_pedido_fabrica": row["valor_pedido_fabrica"],
            }])],
            ignore_index=True,
        )

    out = pd.DataFrame(future_rows)
    out.to_csv(PROCESSED_DATA_DIR / "forecast_future_30d.csv", index=False)

    print(f"Previsão futura gerada: {len(out)} dias")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


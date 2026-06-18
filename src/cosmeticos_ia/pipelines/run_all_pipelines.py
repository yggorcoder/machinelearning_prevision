from cosmeticos_ia.data.run_data_pipeline import main as data_main
from cosmeticos_ia.models.run_ml_pipeline import main as ml_main
from cosmeticos_ia.models.train_forecast import main as forecast_train_main
from cosmeticos_ia.models.backtest_forecast import main as forecast_backtest_main
from cosmeticos_ia.models.select_forecast_model import main as forecast_select_main
from cosmeticos_ia.models.predict_forecast import main as forecast_predict_main
from cosmeticos_ia.models.monitoring import main as monitoring_main


def main() -> None:
    print("[1/7] Data pipeline")
    data_main()

    print("\n[2/7] ML Propensão (treino + scoring + Recall@K + uplift)")
    ml_main()

    print("\n[3/7] Forecast — treino com baselines")
    forecast_train_main()

    print("\n[4/7] Forecast — backtest walk-forward")
    forecast_backtest_main()

    print("\n[5/7] Forecast — seleção de modelo de produção")
    forecast_select_main()

    print("\n[6/7] Forecast — previsão futura (30d)")
    forecast_predict_main(horizon_days=30)

    print("\n[7/7] Monitoramento de drift")
    monitoring_main()

    print("\nPipeline geral concluído.")


if __name__ == "__main__":
    main()

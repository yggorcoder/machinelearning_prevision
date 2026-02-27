from cosmeticos_ia.data.run_data_pipeline import main as data_main
from cosmeticos_ia.models.run_ml_pipeline import main as ml_main
from cosmeticos_ia.models.train_forecast import main as forecast_train_main
from cosmeticos_ia.models.backtest_forecast import main as forecast_backtest_main
from cosmeticos_ia.models.select_forecast_model import main as forecast_select_main
from cosmeticos_ia.models.predict_forecast import main as forecast_predict_main


def main() -> None:
    print("[1/6] Data pipeline")
    data_main()

    print("\n[2/6] ML Propensão")
    ml_main()

    print("\n[3/6] Forecast train")
    forecast_train_main()

    print("\n[4/6] Forecast backtest")
    forecast_backtest_main()

    print("\n[5/6] Seleção de modelo de produção (forecast)")
    forecast_select_main()

    print("\n[6/6] Forecast futuro (30d)")
    forecast_predict_main(horizon_days=30)

    print("\nPipeline geral concluído.")


if __name__ == "__main__":
    main()

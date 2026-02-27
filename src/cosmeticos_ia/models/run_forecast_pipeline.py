from cosmeticos_ia.models.train_forecast import main as train_forecast_main
from cosmeticos_ia.models.predict_forecast import main as predict_forecast_main


def main() -> None:
    print("[1/2] Treinando modelo de previsão...")
    train_forecast_main()

    print("\n[2/2] Gerando previsão futura (30 dias)...")
    predict_forecast_main(horizon_days=30)

    print("\nPipeline de forecast concluído com sucesso.")


if __name__ == "__main__":
    main()

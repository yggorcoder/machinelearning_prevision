from __future__ import annotations

from cosmeticos_ia.models.train import main as train_main
from cosmeticos_ia.models.predict import main as predict_main
from cosmeticos_ia.models.evaluate_campaign import main as evaluate_main


def main() -> None:
    print("[1/3] Treinando modelo...")
    train_main()

    print("\n[2/3] Gerando scoring e campanha...")
    predict_main()

    print("\n[3/3] Avaliando Recall@K...")
    evaluate_main()

    print("\nPipeline de ML concluido com sucesso.")


if __name__ == "__main__":
    main()

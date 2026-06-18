from __future__ import annotations

from cosmeticos_ia.models.train import main as train_main
from cosmeticos_ia.models.predict import main as predict_main
from cosmeticos_ia.models.evaluate_campaign import main as evaluate_main
from cosmeticos_ia.models.evaluate_campaign_uplift import main as uplift_main


def main() -> None:
    print("[1/4] Treinando modelo de propensão...")
    train_main()

    print("\n[2/4] Gerando scoring e campanha Top-100...")
    predict_main()

    print("\n[3/4] Avaliando Recall@K por snapshot...")
    evaluate_main()

    print("\n[4/4] Calculando uplift: modelo vs baseline de recency...")
    uplift_main()

    print("\nPipeline de ML concluído.")


if __name__ == "__main__":
    main()

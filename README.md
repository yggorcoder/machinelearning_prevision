# Cosméticos IA

Projeto de Machine Learning aplicado à gestão comercial e operacional de uma distribuidora de cosméticos.

Machine Learning project for commercial and operational decision-making in a cosmetics distribution business.

## PT-BR

### 1. Visão Geral
Este projeto apoia duas decisões principais do negócio:
- Reativação e priorização de clientes.
- Planejamento de faturamento/compras com previsão de demanda.

Status atual:
- Pipeline de dados com checagem de qualidade.
- Modelo de propensão (ranking comercial + campanha Top50).
- Modelo de forecast com seleção de produção por backtest.
- Dashboard executivo em Streamlit.

### 2. Objetivos de Negócio
- Aumentar receita com ação comercial orientada por score.
- Melhorar cobertura de clientes compradores com listas semanais.
- Reduzir erro de previsão para decisão operacional.

### 3. Fontes de Dados
Arquivos brutos (`data/raw/`):
- `clientes.xlsx`
- `compras_clientes.xlsx`
- `pedidos_cd.xlsx`

### 4. Estrutura do Projeto
```text
cosmeticos-ia/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ docs/
├─ notebooks/
├─ src/cosmeticos_ia/
│  ├─ app/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  └─ pipelines/
├─ tests/
└─ requirements.txt
```

### 5. Setup do Ambiente
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 6. Execução Oficial (1 comando)
Na raiz do projeto:
```powershell
$env:PYTHONPATH="src"
python -m cosmeticos_ia.pipelines.run_all_pipelines
```

Este comando executa em sequência:
1. `data.run_data_pipeline`
2. `models.run_ml_pipeline` (propensão)
3. `models.train_forecast`
4. `models.backtest_forecast`
5. `models.select_forecast_model`
6. `models.predict_forecast`

### 6.1 Dados Fake para GitHub (sem expor dados reais)
Gerar planilhas fake com o mesmo schema:
```powershell
$env:PYTHONPATH="src"
python -m cosmeticos_ia.data.generate_fake_raw
```

Arquivos gerados em `data/raw_fake/`.

Para rodar o pipeline usando os dados fake:
```powershell
$env:PYTHONPATH="src"
$env:COSMETICOS_RAW_DATA_DIR="data/raw_fake"
python -m cosmeticos_ia.pipelines.run_all_pipelines
```

### 7. Artefatos Gerados (data/processed)
Dados e qualidade:
- `quality_report.csv`
- `compras_clientes_clean.parquet`
- `clientes_clean.parquet`
- `pedidos_cd_clean.parquet`
- `kpis_diarios.parquet`
- `kpis_diarios_features.parquet`

Propensão:
- `propensity_model.joblib`
- `propensity_metrics.csv`
- `propensity_scoring.csv`
- `campanha_top50.csv`
- `propensity_recall_at_k_by_snapshot.csv`

Forecast:
- `forecast_model_comparison.csv`
- `forecast_backtest_detail.csv`
- `forecast_backtest_summary.csv`
- `forecast_model_selection.csv`
- `forecast_model.joblib` (modelo de produção)
- `forecast_metrics.csv`
- `forecast_predictions_test.csv`
- `forecast_future_30d.csv`

### 8. Métricas-Chave
Propensão:
- `PR-AUC`, `ROC-AUC`
- `Recall@K por snapshot` (métrica operacional)

Forecast:
- `MAE`, `MAPE`, `WAPE`
- Seleção final de produção baseada em backtest (estabilidade)

### 9. Dashboard
Rodar:
```powershell
$env:PYTHONPATH="src"
streamlit run src/cosmeticos_ia/app/dashboard.py
```

### 10. Rotina Semanal Recomendada
1. Atualizar arquivos brutos em `data/raw/`.
2. Rodar `run_all_pipelines`.
3. Revisar `forecast_model_selection.csv` e métricas.
4. Executar campanha com `campanha_top50.csv`.
5. Acompanhar resultados no dashboard.

### 11. Testes
```powershell
$env:PYTHONPATH="src"
pytest -q
```

---

## EN

### 1. Overview
This project supports two core business decisions:
- Customer reactivation and prioritization.
- Revenue/purchase planning through demand forecasting.

Current status:
- Data pipeline with quality checks.
- Propensity model (commercial ranking + Top50 campaign).
- Forecast model with production selection via backtesting.
- Executive dashboard in Streamlit.

### 2. Business Goals
- Increase revenue using score-driven commercial actions.
- Improve buyer coverage with weekly ranked lists.
- Reduce forecast error for operational planning.

### 3. Data Sources
Raw files (`data/raw/`):
- `clientes.xlsx`
- `compras_clientes.xlsx`
- `pedidos_cd.xlsx`

### 4. Project Structure
```text
cosmeticos-ia/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ docs/
├─ notebooks/
├─ src/cosmeticos_ia/
│  ├─ app/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  └─ pipelines/
├─ tests/
└─ requirements.txt
```

### 5. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 6. Official Run (single command)
From project root:
```powershell
$env:PYTHONPATH="src"
python -m cosmeticos_ia.pipelines.run_all_pipelines
```

This command executes:
1. `data.run_data_pipeline`
2. `models.run_ml_pipeline` (propensity)
3. `models.train_forecast`
4. `models.backtest_forecast`
5. `models.select_forecast_model`
6. `models.predict_forecast`

### 6.1 Fake Data for GitHub (without exposing real data)
Generate fake spreadsheets with the same schema:
```powershell
$env:PYTHONPATH="src"
python -m cosmeticos_ia.data.generate_fake_raw
```

Files are written to `data/raw_fake/`.

To run the pipeline with fake data:
```powershell
$env:PYTHONPATH="src"
$env:COSMETICOS_RAW_DATA_DIR="data/raw_fake"
python -m cosmeticos_ia.pipelines.run_all_pipelines
```

### 7. Generated Artifacts (data/processed)
Data and quality:
- `quality_report.csv`
- `compras_clientes_clean.parquet`
- `clientes_clean.parquet`
- `pedidos_cd_clean.parquet`
- `kpis_diarios.parquet`
- `kpis_diarios_features.parquet`

Propensity:
- `propensity_model.joblib`
- `propensity_metrics.csv`
- `propensity_scoring.csv`
- `campanha_top50.csv`
- `propensity_recall_at_k_by_snapshot.csv`

Forecast:
- `forecast_model_comparison.csv`
- `forecast_backtest_detail.csv`
- `forecast_backtest_summary.csv`
- `forecast_model_selection.csv`
- `forecast_model.joblib` (production model)
- `forecast_metrics.csv`
- `forecast_predictions_test.csv`
- `forecast_future_30d.csv`

### 8. Key Metrics
Propensity:
- `PR-AUC`, `ROC-AUC`
- `Recall@K by snapshot` (operational metric)

Forecast:
- `MAE`, `MAPE`, `WAPE`
- Final production selection based on backtest stability

### 9. Dashboard
```powershell
$env:PYTHONPATH="src"
streamlit run src/cosmeticos_ia/app/dashboard.py
```

### 10. Recommended Weekly Routine
1. Update raw files in `data/raw/`.
2. Run `run_all_pipelines`.
3. Review `forecast_model_selection.csv` and metrics.
4. Execute campaign using `campanha_top50.csv`.
5. Track outcomes in dashboard.

### 11. Tests
```powershell
$env:PYTHONPATH="src"
pytest -q
```

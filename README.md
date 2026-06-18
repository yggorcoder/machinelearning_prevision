# Cosméticos IA

[![CI](https://github.com/yggorcoder/machinelearning_prevision/actions/workflows/ci.yml/badge.svg)](https://github.com/yggorcoder/machinelearning_prevision/actions/workflows/ci.yml)

Machine Learning project for commercial and operational decision-making in a cosmetics distribution business.

---

## PT-BR

### Resultados (dados reais — `data/raw`, periodo de teste out/2025)

> Reproduza com: `python -m cosmeticos_ia.pipelines.run_all_pipelines`  
> Com dados sinteticos (CI/GitHub): `$env:COSMETICOS_RAW_DATA_DIR="data/raw_fake"`

#### Modelo de propensão de recompra

| Estratégia | PR-AUC | Recall@50 | Precisão@50 |
|------------|--------|-----------|-------------|
| Baseline (recência) | 0,42 | 1,5% | 52% |
| **Random Forest + RFM** | **0,54** | **2,8%** | **94%** |

- Ganho PR-AUC vs baseline: **+12,1 p.p.**
- Uplift Recall@50 vs baseline: **+4,2 p.p.** | Lift@50: **1,51×** *(métricas de avaliação no Top 50; a lista operacional exportada usa **Top 100**)*

#### Forecast de faturamento diário (walk-forward backtest, 9 folds)

| Modelo | WAPE médio | MAE médio |
|--------|------------|-----------|
| Baseline lag-7 | 82,5% | R$ 687/dia |
| Baseline média móvel 7d | 62,9% | R$ 532/dia |
| **Random Forest (produção)** | **7,6%** | **R$ 65/dia** |

- Erro no split atual de validação: **WAPE 8,6%** (MAE R$ 76/dia)
- Ganho WAPE vs melhor baseline no split: **−58,6 p.p.**

> Valores gerados automaticamente pelo pipeline. Dados proprietários não são versionados no GitHub.

---

### 1. Visão Geral

Este projeto apoia duas decisões principais do negócio:

- **Reativação e priorização de clientes** — lista semanal **Top 100** com score de propensão (exportável em CSV).
- **Planejamento de faturamento/compras** — previsão de demanda 30 dias à frente.

**Status atual:**
- Pipeline de dados com checagem de qualidade e bloqueio em erro crítico.
- Modelo de propensão com baseline de recency, validação temporal e avaliação de uplift.
- Modelo de forecast com baselines naives, seleção por backtest walk-forward.
- Monitoramento de drift de features e faturamento.
- Dashboard executivo em Streamlit.
- CI/CD via GitHub Actions (testes + smoke test do pipeline completo).

### 2. Objetivos de Negócio

- Aumentar receita com ação comercial orientada por score.
- Melhorar cobertura de clientes compradores com listas semanais.
- Reduzir erro de previsão para decisão operacional de compras.

### 3. Fontes de Dados

**Dados reais (apenas local, não versionados)** — pasta `data/raw/`:
- `clientes.xlsx`
- `compras_clientes.xlsx`
- `pedidos_cd.xlsx`

**Dados sintéticos (versionados no GitHub)** — pasta `data/raw_fake/`:
- Mesmos nomes de arquivo, com nomes, CPFs, e-mails e valores fictícios gerados por `generate_fake_raw`.
- Usados na CI e para quem clonar o repositório sem acesso aos dados proprietários.

Consulte [`docs/data_dictionary.md`](docs/data_dictionary.md) para descrição completa de todas as colunas e métricas.

#### Privacidade — o que não vai para o GitHub

| Item | Versionado? |
|------|-------------|
| `data/raw/` (dados reais) | Não |
| `data/processed/` (parquets, modelos, campanhas) | Não |
| `notebooks/` (EDA local) | Não |
| `.env` e credenciais | Não |
| `data/raw_fake/` (dados sintéticos) | Sim |

### 4. Estrutura do Projeto

```text
cosmeticos-ia/
├─ .github/workflows/ci.yml     ← CI: testes + pipeline smoke test
├─ data/
│  ├─ raw/                      ← dados proprietários (não versionados)
│  ├─ raw_fake/                 ← dados sintéticos para CI/reprodutibilidade
│  └─ processed/                ← artefatos gerados (não versionados)
├─ docs/
│  ├─ data_dictionary.md        ← dicionário de dados completo
│  ├─ 01_escopo_negocio.md
│  ├─ 02_metricas_e_targets.md
│  └─ 03_plano_execucao.md
├─ notebooks/                   ← EDA local (não versionado; pode conter dados reais)
├─ src/cosmeticos_ia/
│  ├─ app/dashboard.py          ← 5 abas: Geral, Campanhas, Forecast, Uplift, Monitoramento
│  ├─ data/           ← loaders, preprocessing, quality, fake data
│  ├─ features/       ← build_features, build_training_data, geocode
│  ├─ models/
│  │  ├─ metrics.py                    ← métricas compartilhadas (WAPE, PR-AUC, Recall@K…)
│  │  ├─ train.py                      ← propensão: RF + baseline recency
│  │  ├─ predict.py                    ← scoring + campanha Top-100
│  │  ├─ evaluate_campaign.py          ← Recall@K por snapshot
│  │  ├─ evaluate_campaign_uplift.py   ← uplift modelo vs baseline
│  │  ├─ train_forecast.py             ← forecast: RF/XGB + baselines naives
│  │  ├─ backtest_forecast.py          ← walk-forward backtest
│  │  ├─ select_forecast_model.py      ← seleção por WAPE médio
│  │  ├─ predict_forecast.py           ← previsão futura 30d
│  │  ├─ run_ml_pipeline.py            ← orquestrador ML
│  │  └─ monitoring.py                 ← drift de features e faturamento
│  └─ pipelines/run_all_pipelines.py   ← orquestrador completo (7 etapas)
└─ tests/
   ├─ data/test_data_quality.py
   ├─ features/test_geocode.py
   └─ models/
      ├─ test_metrics.py               ← métricas compartilhadas
      ├─ test_build_training_data.py   ← invariantes do dataset de propensão
      └─ test_uplift.py                ← avaliação de uplift de campanha
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

1. `data.run_data_pipeline` — xlsx → parquet, qualidade + features
2. `models.run_ml_pipeline` — propensão + scoring + Recall@K + uplift
3. `models.train_forecast` — treino com baselines naives
4. `models.backtest_forecast` — walk-forward backtest
5. `models.select_forecast_model` — seleção por WAPE médio
6. `models.predict_forecast` — previsão 30 dias
7. `models.monitoring` — relatório de drift

### 6.1 Dados Fake para GitHub (sem expor dados reais)

```powershell
$env:PYTHONPATH="src"
python -m cosmeticos_ia.data.generate_fake_raw
```

Arquivos gerados em `data/raw_fake/`.

Para rodar o pipeline com dados fake:

```powershell
$env:PYTHONPATH="src"
$env:COSMETICOS_RAW_DATA_DIR="data/raw_fake"
python -m cosmeticos_ia.pipelines.run_all_pipelines
```

### 7. Artefatos Gerados (`data/processed/`)

**Dados e qualidade:**
- `quality_report.csv`
- `compras_clientes_clean.parquet`, `clientes_clean.parquet`, `pedidos_cd_clean.parquet`
- `kpis_diarios.parquet`, `kpis_diarios_features.parquet`

**Propensão:**
- `propensity_model.joblib`
- `propensity_metrics.csv` — PR-AUC, ROC-AUC, Recall@K, Precision@K para modelo e baseline
- `propensity_scoring.csv` — todos os clientes ranqueados
- `campanha_top100.csv` — lista operacional da semana (Top 100)
- `propensity_recall_at_k_by_snapshot.csv`
- `campaign_uplift_summary.csv` — uplift por estratégia e K
- `campaign_uplift_vs_baseline.csv` — ganho em p.p. vs baseline

**Forecast:**
- `forecast_model_comparison.csv` — todos os modelos + uplift vs naive
- `forecast_backtest_detail.csv`, `forecast_backtest_summary.csv`
- `forecast_model_selection.csv` — modelo eleito para produção
- `forecast_model.joblib`
- `forecast_metrics.csv`
- `forecast_predictions_test.csv`
- `forecast_future_30d.csv`

**Monitoramento:**
- `monitoring_report.csv` — PSI e mean shift por feature
- `monitoring_summary.csv` — status geral (OK / WARNING / ALERT)

### 8. Métricas-Chave

**Propensão:**
- `PR-AUC` (principal — dados desbalanceados)
- `Recall@K` e `Precision@K` por snapshot (operacional)
- `Uplift@K` vs baseline de recency (negócio)

**Forecast:**
- `WAPE` (principal — robusto a zeros)
- `MAE`, `MAPE`
- Seleção de produção baseada em estabilidade de backtest

**Monitoramento:**
- `PSI` por feature — > 0.2 dispara ALERT de retreino
- `mean_shift_normalized` — desvio de médias em σ

### 9. Dashboard

Cinco abas: **Visão Geral**, **Campanhas**, **Forecast**, **Uplift**, **Monitoramento**.

```powershell
$env:PYTHONPATH="src"
streamlit run src/cosmeticos_ia/app/dashboard.py
```

> Rode o pipeline antes para gerar os artefatos em `data/processed/`.

### 10. Rotina Semanal Recomendada

1. Atualizar arquivos brutos em `data/raw/`.
2. Rodar `run_all_pipelines`.
3. Verificar `monitoring_summary.csv` — se ALERT, retreinar.
4. Revisar `forecast_model_selection.csv` e métricas.
5. Executar campanha com `campanha_top100.csv`.
6. Acompanhar resultados no dashboard.

### 11. Testes

```powershell
$env:PYTHONPATH="src"
pytest -q
```

Cobertura dos testes unitários:
- Qualidade de dados (`tests/data/`)
- Geocodificação (`tests/features/`)
- Métricas compartilhadas — WAPE, PR-AUC, Recall@K, splits temporais (`tests/models/test_metrics.py`)
- Invariantes do dataset de propensão — sem leakage, target binário, ordem temporal (`tests/models/test_build_training_data.py`)
- Uplift de campanha — lift, recall monotônico, limites (`tests/models/test_uplift.py`)

---

## EN

### 1. Overview

This project supports two core business decisions:

- **Customer reactivation and prioritization** — weekly **Top 100** list scored by repurchase propensity.
- **Revenue/purchase planning** — 30-day demand forecasting.

**Current status:**
- Data pipeline with quality checks and critical-error blocking.
- Propensity model with recency baseline, temporal validation, and uplift evaluation.
- Forecast model with naive baselines, production selection via walk-forward backtesting.
- Feature and revenue drift monitoring.
- Executive dashboard in Streamlit.
- CI/CD via GitHub Actions (unit tests + full pipeline smoke test).

### 2. Business Goals

- Increase revenue using score-driven commercial actions.
- Improve buyer coverage with weekly ranked lists.
- Reduce forecast error for operational planning.

### 3. Data Sources

**Real data (local only, not versioned)** — `data/raw/`:
- `clientes.xlsx`, `compras_clientes.xlsx`, `pedidos_cd.xlsx`

**Synthetic data (versioned on GitHub)** — `data/raw_fake/`:
- Same schema, anonymized values for CI and public reproducibility.

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for full column and metric descriptions.

**Not versioned:** `data/raw/`, `data/processed/`, `notebooks/`, `.env`.

### 4. Project Structure

See PT-BR section above (structure is language-agnostic).

### 5. Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 6. Official Run (single command)

```powershell
$env:PYTHONPATH="src"
python -m cosmeticos_ia.pipelines.run_all_pipelines
```

Executes 7 steps: data pipeline → propensity (train + score + Recall@K + uplift) → forecast (train + backtest + select + predict) → monitoring.

### 6.1 Fake Data for GitHub

```powershell
$env:PYTHONPATH="src"
python -m cosmeticos_ia.data.generate_fake_raw
# then:
$env:COSMETICOS_RAW_DATA_DIR="data/raw_fake"
python -m cosmeticos_ia.pipelines.run_all_pipelines
```

### 7. Generated Artifacts

See PT-BR section — artifact names are the same.

### 8. Key Metrics

**Propensity:** PR-AUC, Recall@K / Precision@K by snapshot, Lift@K vs recency baseline.

**Forecast:** WAPE (primary), MAE, MAPE, production selection by backtest stability.

**Monitoring:** PSI per feature (> 0.2 = ALERT), normalised mean shift.

### 9. Dashboard

Five tabs: Overview, Campaigns, Forecast, Uplift, Monitoring.

```powershell
$env:PYTHONPATH="src"
streamlit run src/cosmeticos_ia/app/dashboard.py
```

> Run the pipeline first to generate artifacts in `data/processed/`.

### 10. Recommended Weekly Routine

1. Update raw files in `data/raw/`.
2. Run `run_all_pipelines`.
3. Check `monitoring_summary.csv` — retrain if ALERT.
4. Review `forecast_model_selection.csv` and metrics.
5. Execute campaign using `campanha_top100.csv`.
6. Track outcomes in the dashboard.

### 11. Tests

```powershell
$env:PYTHONPATH="src"
pytest -q
```

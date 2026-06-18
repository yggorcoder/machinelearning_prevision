# Data Dictionary — Cosméticos IA

## Fontes brutas (`data/raw/`)

### `clientes.xlsx`
| Coluna original   | Coluna padronizada     | Tipo      | Descrição                                   |
|-------------------|------------------------|-----------|---------------------------------------------|
| Codigo            | id_cliente             | str       | Identificador único do cliente              |
| Nome              | nome_cliente           | str       | Nome do cliente/revendedor                  |
| Cod Master        | id_master              | str       | ID do upline (quem indicou)                 |
| Nome Master       | nome_master            | str       | Nome do upline                              |
| Dt Cadastro       | data_cadastro          | datetime  | Data de inclusão na base                    |
| Dt Ultima Compra  | data_ultima_compra     | datetime  | Data da compra mais recente (snapshot CRM)  |
| Qtde Itens        | qtde_itens             | int       | Quantidade de itens comprados (acumulado)   |
| Qtde Pontos Total | qtde_pontos_total      | float     | Pontuação acumulada no programa de fidelidade |
| Qtde Pontos Mes   | qtde_pontos_mes        | float     | Pontos no mês corrente                      |
| Qtde Ativacoes Mes| qtde_ativacoes_mes     | int       | Ativações de downline no mês                |

### `compras_clientes.xlsx`
| Coluna original   | Coluna padronizada     | Tipo      | Descrição                                   |
|-------------------|------------------------|-----------|---------------------------------------------|
| OP                | id_pedido              | str       | Número do pedido (OP)                       |
| Pedido            | pedido                 | str       | Código interno do pedido                    |
| Origem            | origem                 | str       | Canal de origem (app, televendas, etc.)     |
| Primeira OP       | primeira_op            | str       | Indica se é a primeira compra do cliente    |
| Cód. Cliente      | id_cliente             | str       | FK → clientes.id_cliente                   |
| Nome              | nome                   | str       | Nome do cliente na compra                   |
| Nível             | nivel                  | str       | Nível do cliente no momento da compra       |
| Qtde Itens        | qtde_itens             | int       | Quantidade de itens no pedido               |
| Valor Total       | valor_total            | float     | Valor total do pedido (R$)                  |
| Valor Desconto    | valor_desconto         | float     | Desconto aplicado (R$)                      |
| Data              | data_compra            | datetime  | Data da compra                              |

### `pedidos_cd.xlsx`
| Coluna original   | Coluna padronizada     | Tipo      | Descrição                                           |
|-------------------|------------------------|-----------|-----------------------------------------------------|
| Pedido            | id_pedido              | str       | Número do pedido do CD para a fábrica               |
| Origem Pedido     | origem_pedido          | str       | Origem do pedido (site, representante, etc.)        |
| Nome Consultor    | nome_cliente           | str       | Nome do consultor/responsável                       |
| Valor             | valor_pedido           | float     | Valor do pedido ao fornecedor (R$)                  |
| Data Pedido       | data_pedido            | datetime  | Data de emissão do pedido                           |
| Data Nota Fiscal  | data_nota_fiscal       | datetime  | Data de emissão da NF de saída                      |

---

## Dados processados (`data/processed/`)

### KPIs diários — `kpis_diarios.parquet`
| Coluna               | Tipo    | Descrição                                        |
|----------------------|---------|--------------------------------------------------|
| data                 | date    | Data de referência                               |
| faturamento_total    | float   | Soma dos valores de compras no dia               |
| numero_pedidos       | int     | Pedidos únicos no dia                            |
| clientes_unicos      | int     | Clientes distintos que compraram no dia          |
| ticket_medio         | float   | faturamento_total / numero_pedidos               |
| valor_pedido_fabrica | float   | Valor dos pedidos CD→fábrica no dia              |

### Dataset de propensão — `propensity_dataset.parquet`
| Coluna                   | Tipo    | Descrição                                                          |
|--------------------------|---------|--------------------------------------------------------------------|
| id_cliente               | str     | Identificador do cliente                                           |
| snapshot_date            | date    | Data de referência do snapshot semanal                             |
| recency                  | int     | Dias desde a última compra até snapshot_date                       |
| frequency_lookback       | int     | Pedidos únicos nos últimos 90 dias até snapshot_date               |
| monetary_lookback        | float   | Valor total comprado nos últimos 90 dias                           |
| ticket_medio_lookback    | float   | monetary_lookback / frequency_lookback                             |
| tendencia_gasto_lookback | float   | monetary_30d / monetary_90d — acelera/desacelera gasto recente (0–1) |
| target_recompra_30d      | int     | 1 se cliente comprou nos 30 dias seguintes; 0 caso contrário       |

---

## Artefatos de modelos

### `propensity_metrics.csv`
| Coluna                     | Descrição                                               |
|----------------------------|---------------------------------------------------------|
| model                      | Nome do modelo (random_forest / baseline_recency)       |
| pr_auc                     | Área sob a curva Precisão-Recall                        |
| roc_auc                    | Área sob a curva ROC                                    |
| recall_at_10/30/50         | Recall@K no conjunto de teste                           |
| precision_at_10/30/50      | Precision@K no conjunto de teste                        |
| uplift_pr_auc_vs_baseline  | Ganho de PR-AUC vs baseline (apenas linha do modelo)    |

### `campaign_uplift_vs_baseline.csv`
| Coluna               | Descrição                                                  |
|----------------------|------------------------------------------------------------|
| k                    | Tamanho da lista (10, 30, 50)                              |
| recall_uplift_pp     | Ganho de Recall@K em pontos percentuais vs baseline        |
| precision_uplift_pp  | Ganho de Precision@K em pontos percentuais vs baseline     |
| lift_model           | Lift do modelo (Precision@K / taxa de compra global)       |
| lift_baseline        | Lift do baseline                                           |

### `forecast_backtest_summary.csv`
| Coluna          | Descrição                                                     |
|-----------------|---------------------------------------------------------------|
| model           | Nome (naive_lag7 / naive_rolling7 / random_forest / xgboost) |
| folds           | Número de folds de backtest                                   |
| mae_mean/std    | MAE médio e desvio entre folds                               |
| mape_mean/std   | MAPE médio e desvio entre folds                              |
| wape_mean/std   | WAPE médio e desvio entre folds                              |
| wape_uplift_vs_best_baseline | Redução de WAPE em relação ao melhor baseline naive |

### `monitoring_report.csv`
| Coluna                   | Descrição                                                     |
|--------------------------|---------------------------------------------------------------|
| feature                  | Nome da feature monitorada                                    |
| ref_mean / ref_std       | Estatísticas no período de treino                             |
| current_mean / current_std | Estatísticas na janela recente                              |
| mean_shift_normalized    | Desvio de média normalizado pelo desvio-padrão de referência  |
| psi                      | Population Stability Index                                    |
| status                   | OK / WARNING / ALERT                                          |

---

## Glossário de métricas

| Métrica        | Fórmula / Definição                                                           | Uso                          |
|----------------|-------------------------------------------------------------------------------|------------------------------|
| PR-AUC         | Área sob curva Precisão × Recall                                              | Propensão (dados desbalanceados) |
| ROC-AUC        | Área sob curva TPR × FPR                                                      | Propensão (referência)       |
| Recall@K       | Compradores capturados no Top-K / total de compradores reais                  | Operacional: cobertura        |
| Precision@K    | Compradores no Top-K / K                                                      | Operacional: taxa de acerto   |
| Lift@K         | Precision@K / taxa base de compradores                                        | Eficiência vs aleatório       |
| MAE            | Média dos erros absolutos                                                     | Forecast                     |
| MAPE           | Média dos erros percentuais (ignora zeros)                                    | Forecast                     |
| WAPE           | ∑|y - ŷ| / ∑|y|  — robusto a séries com zeros                               | Forecast (principal)         |
| PSI            | Population Stability Index — mede drift de distribuição                       | Monitoramento                |

# Métricas e Targets de ML

## Modelo 1: Propensão de recompra (reativação/churn)

### Pergunta
Quem tem maior probabilidade de comprar nos próximos 30 dias?

### Unidade de modelagem
Cliente-data de referência (snapshot semanal)

### Target (y)
y = 1 se cliente realizar >=1 compra nos 30 dias após a data de referência; senão 0.

### Principais features candidatas
- RFM (recency, frequency, monetary)
- Tendência recente (gasto últimos 30/60/90 dias)
- Relação com CD (cliente deste CD/outro)
- Status cadastral
- Sazonalidade (mês, semana)

### Métricas técnicas
- PR-AUC (principal)
- Recall@TopK (principal para operação)
- ROC-AUC (secundária)

### Métrica de negócio
Receita gerada por lista TopK vs baseline atual.

## Modelo 2: Previsão de faturamento diário/semanal

### Pergunta
Qual o faturamento esperado nas próximas 4 a 8 semanas?

### Série alvo
faturamento_total diário (ou semanal agregado)

### Métricas técnicas
- MAE
- MAPE
- WAPE

### Métrica de negócio
Erro de planejamento reduzido e menor ruptura/excesso.

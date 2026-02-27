# Plano de Execução

## Fase A - Dados e qualidade
- Pipeline único: raw -> preprocess -> features -> datasets de modelo
- Validações de qualidade (tipos, nulos, duplicados, datas)
- Data dictionary

## Fase B - Modelo de propensão
- Construção de dataset supervisionado temporal
- Baseline + modelo principal
- Ranking acionável de clientes

## Fase C - Modelo de previsão
- Baseline temporal + modelo principal
- Backtesting por janelas de tempo
- Previsão 4-8 semanas com intervalo

## Fase D - Produto
- Dashboard executivo
- Exportação de listas de ação
- Rotina semanal de atualização

## Fase E - Governança
- Testes automatizados
- README técnico + executivo
- Critérios de monitoramento pós-deploy

import pandas as pd

def build_rfm(
        df_compras: pd.DataFrame,
        df_clientes: pd.DataFrame | None = None,
        reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Calcula RFM (Recency, Frequency, Monetary) por cliente.

    Espera que df_compras já tenha:
      - id_cliente (str)
      - id_pedido  (str)
      - data_compra (datetime)
      - valor_total (float)
    """
    df = df_compras.copy()

    df["data_compra"] = pd.to_datetime(df["data_compra"])
    df["id_cliente"] = df["id_cliente"].astype(str)

    if reference_date is None:
        reference_date = df["data_compra"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("id_cliente").agg(
        ultima_compra=("data_compra", "max"),
        frequency=("id_pedido", "nunique"), #num. pedidos distintos
        monetary=("valor_total", "sum"),
    )

    rfm["recency"] = (reference_date - rfm["ultima_compra"]).dt.days
    rfm = rfm.reset_index()

    #junta com base de clinetes, se disponível
    if df_clientes is not None and "id_cliente" in df_clientes.columns:
        dfc = df_clientes.copy()
        dfc["id_cliente"] = dfc["id_cliente"].astype(str)
        rfm = rfm.merge(dfc, on="id_cliente", how="left")
    
    return rfm

def build_daily_kpis(df_compras: pd.DataFrame, df_pedidos: pd.DataFrame) -> pd.DataFrame:
    """
    Gera indicadores diários do CD:
      - faturamento_total
      - numero_pedidos
      - clientes_unicos
      - ticket_medio
      - valor_pedido_fabrica (pedidos CD → fábrica)
    """
    dfc = df_compras.copy()
    dfp = df_pedidos.copy()

    dfc["data_compra"] = pd.to_datetime(dfc["data_compra"])
    dfp["data_pedido"] = pd.to_datetime(dfp["data_pedido"])

    # vendas para clientes
    vendas_daily = (
        dfc.groupby(dfc["data_compra"].dt.date)
        .agg(
            faturamento_total=("valor_total", "sum"),
            numero_pedidos=("id_pedido", "nunique"),
            clientes_unicos=("id_cliente", "nunique"),
        )
        .rename_axis("data")
    )

    vendas_daily["ticket_medio"] = (
        vendas_daily["faturamento_total"] / vendas_daily["numero_pedidos"]
    )

    # pedidos do cd para a fábrica
    if "valor_pedido" in dfp.columns:
        pedidos_daily = (
            dfp.groupby(dfp["data_pedido"].dt.date)
            .agg(valor_pedido_fabrica=("valor_pedido", "sum"))
            .rename_axis("data")
        )
    else:
        pedidos_daily = pd.DataFrame(columns=["valor_pedido_fabrica"])

    # junta tudo
    daily = vendas_daily.join(pedidos_daily, how="left").fillna({"valor_pedido_fabrica": 0})
    daily.index = pd.to_datetime(daily.index)

    return daily.sort_index()
import pandas as pd

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas para minúsculo, com _ e sem acentos."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
        .str.replace("á", "a")
        .str.replace("ã", "a")
        .str.replace("à", "a")
        .str.replace("é", "e")
        .str.replace("ê", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("õ", "o")
        .str.replace("ú", "u")
        .str.replace("ç", "c")
    
    )
    return df

def _parse_money_brl(series: pd.Series) -> pd.Series:
    """
    Converte valores monetários brasileiros em float, lidando com formatos:
      - "1.234,56"
      - "1234,56"
      - "1234.56"
      - "1234"
    """
    s = series.astype(str).str.strip()

    def _parse_one(x: str):
        if x == "" or x.lower() in {"nan", "none"}:
            return None

        x = x.replace(" ", "")

        # Caso 1: tem ponto e vírgula -> padrão BR "1.234,56"
        if "," in x and "." in x:
            x = x.replace(".", "").replace(",", ".")
        # Caso 2: só vírgula -> trata vírgula como decimal
        elif "," in x:
            x = x.replace(",", ".")
        # Caso 3: só ponto -> assume que já está no padrão decimal
        # Caso 4: só dígitos -> deixa como está

        try:
            return float(x)
        except ValueError:
            return None

    return s.apply(_parse_one).astype(float)





def preprocess_compras(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e padroniza a base de compras.

    Esperado que as colunas originais (já corrigidas na conversão) sejam:
    OP, Pedido, Origem, Primeira OP, Cód. Cliente, Nome, Nível,
    Qtde Itens, Valor Total, Valor Desconto, Data
    """
    df = normalize_columns(df)

    # renomeia para nomes padrão do projeto
    rename_map = {
        "cod_cliente": "id_cliente",
        "op": "id_pedido",
        "data": "data_compra",
        "Nivel": "nivel",
        "Qtde Itens": "qtde_itens",
        "Valor Total": "valor_total",
        "Valor Desconto": "valor_desconto",
        
    }
    df = df.rename(columns=rename_map)

    # tipos

    df["id_cliente"] = df["id_cliente"].astype(str).str.strip()
    df["id_pedido"] = df["id_pedido"].astype(str).str.strip()
    df["data_compra"] = pd.to_datetime(
        df["data_compra"], 
        errors="coerce", 
        dayfirst=True, )

    #quantidade
    if "qtde_itens" in df.columns:
        df["qtde_itens"] = pd.to_numeric(df["qtde_itens"], errors="coerce").fillna(0).astype(int)

    # Valor total e desconto -> float
    if "valor_total" in df.columns:
        df["valor_total"] = pd.to_numeric(df["valor_total"], errors="coerce").fillna(0.0)

    if "valor_desconto" in df.columns:
        df["valor_desconto"] = pd.to_numeric(df["valor_desconto"], errors="coerce").fillna(0.0)

    # remove linhas inválidas para análise/modelagem
    # compra válida precisa de cliente, pedido e data
    df = df.dropna(subset=["id_cliente", "id_pedido", "data_compra"]).copy()

    #remove ids vazios após padronização
    df = df[
        (df["id_cliente"].astype(str).str.strip() != "")
        & (df["id_pedido"].astype(str).str.strip() != "")
    ].copy()


    return df

def preprocess_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e padroniza a base de clientes.

    Ajuste o rename_map abaixo para bater com os nomes reais das colunas.
    """
    df = normalize_columns(df)

    # EXEMPLO de mapeamento – adapte aos seus nomes reais:
    rename_map = {
        "cod_master": "id_master",                 # master = quem indicou
        "nome_master": "nome_master",
        "codigo": "id_cliente",
        "nome": "nome_cliente",
        "qtde_itens": "qtde_itens",
        "qtde_equipe": "qtde_equipe",
        "qtde_pontos_total": "qtde_pontos_total",
        "qtde_pontos_no_mes": "qtde_pontos_mes",
        "qtde_ativacoes_no_mes": "qtde_ativacoes_mes",
        "dt_ultima_compra": "data_ultima_compra",
        "dt_cadastro": "data_cadastro",
    }
    # só renomeia quem realmente existir
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if "id_cliente" in df.columns:
        df["id_cliente"] = df["id_cliente"].astype(str).str.strip()

    if "data_ultima_compra" in df.columns:
        df["data_ultima_compra"] = pd.to_datetime(df["data_ultima_compra"], errors="coerce", dayfirst=True)
    if "data_cadastro" in df.columns:
        df["data_cadastro"] = pd.to_datetime(df["data_cadastro"], errors="coerce", dayfirst=True)

    return df

def preprocess_pedidos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e padroniza a base de pedidos do CD para a fábrica.
    """
    df = normalize_columns(df)

    # EXEMPLO – adapte aos seus nomes reais:
    rename_map = {
        "pedido": "id_pedido",
        "origem_pedido": "origem_pedido",          # pode nem precisar renomear
        "nome_consultor": "nome_cliente",          # se o consultor for o "cliente" aqui
        "valor": "valor_pedido",
        "data_pedido": "data_pedido",
        "data_nota_fiscal": "data_nota_fiscal",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if "data_pedido" in df.columns:
        df["data_pedido"] = pd.to_datetime(df["data_pedido"], errors="coerce", dayfirst=True)

    if "valor_pedido" in df.columns:
        df["valor_pedido"] = _parse_money_brl(df["valor_pedido"]).fillna(0.0)

    return df


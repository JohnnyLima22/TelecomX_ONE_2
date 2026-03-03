# data_loader.py

import pandas as pd

def load_data(source_type="csv", path=None):
    """
    Carrega dados de diferentes fontes.
    Se mudar a coleta, alterar apenas aqui.
    """

    if source_type == "csv":
        return pd.read_csv(path)

    elif source_type == "excel":
        return pd.read_excel(path)

    else:
        raise ValueError("Fonte de dados não suportada.")
import pandas as pd
import numpy as np


# ==============================
# 1. DATA SOURCE
# ==============================

def fetch_data(source: str = "url") -> pd.DataFrame:
    """
    Fetch raw data from source.
    Can be extended to support database, API, CSV, etc.
    """

    if source == "url":
        url = "https://raw.githubusercontent.com/alura-cursos/challenge2-data-science/refs/heads/main/TelecomX_Data.json"
        raw = pd.read_json(url)
        df = pd.json_normalize(raw.to_dict(orient="records"))
        return df

    else:
        raise ValueError("Unsupported data source.")


# ==============================
# 2. DATA CLEANING
# ==============================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, type corrections and duplicates.
    """

    # Convert Total Charges to numeric
    df["account.Charges.Total"] = pd.to_numeric(
        df["account.Charges.Total"], errors="coerce"
    )

    # Remove empty churn rows
    df = df[df["Churn"].str.strip() != ""].copy()

    # Remove rows with null total charges
    df.dropna(subset=["account.Charges.Total"], inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


# ==============================
# 3. COLUMN TRANSLATION
# ==============================

def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate column names to PT-BR.
    """

    column_map = {
        "id": "ID_Cliente",
        "Churn": "Evasao",
        "customer.gender": "Genero",
        "customer.SeniorCitizen": "Idoso",
        "customer.Partner": "Parceiro",
        "customer.Dependents": "Dependentes",
        "customer.tenure": "Meses_Contrato_Cliente",
        "phone.PhoneService": "Servico_Telefone",
        "phone.MultipleLines": "Multiplas_Linhas",
        "internet.InternetService": "Servico_Internet",
        "internet.OnlineSecurity": "Seguranca_Online",
        "internet.OnlineBackup": "Backup_Online",
        "internet.DeviceProtection": "Protecao_Dispositivo",
        "internet.TechSupport": "Suporte_Tecnico",
        "internet.StreamingTV": "Streaming_TV",
        "internet.StreamingMovies": "Streaming_Filmes",
        "account.Contract": "Tipo_Contrato",
        "account.PaperlessBilling": "Fatura_Digital",
        "account.PaymentMethod": "Metodo_Pagamento",
        "account.Charges.Monthly": "Valor_Mensal",
        "account.Charges.Total": "Valor_Total",
        "account.tenure": "Meses_Permanencia"
    }

    df.rename(columns=column_map, inplace=True)

    return df


# ==============================
# 4. VALUE STANDARDIZATION
# ==============================

def standardize_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate categorical values to PT-BR.
    """

    value_map = {
        "Yes": "Sim",
        "No": "Nao",
        "Month-to-month": "Mensal",
        "One year": "Anual",
        "Two year": "Bienal",
        "Fiber optic": "Fibra otica",
        "Electronic check": "Cheque eletronico",
        "Mailed check": "Cheque correio",
        "Bank transfer (automatic)": "Transferencia bancaria",
        "Credit card (automatic)": "Cartao de credito",
        "Female": "Feminino",
        "Male": "Masculino"
    }

    df.replace(value_map, inplace=True)

    return df


# ==============================
# 5. BINARY ENCODING
# ==============================

def convert_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns with only 'Sim'/'Nao' to 1/0.
    """

    for col in df.columns:
        unique_values = set(df[col].dropna().unique())

        if unique_values.issubset({"Sim", "Nao"}) and len(unique_values) > 0:
            df[col] = df[col].map({"Sim": 1, "Nao": 0})

    return df


# ==============================
# 6. FEATURE ENGINEERING
# ==============================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new business-relevant features.
    """

    df["Custo_Diario"] = (df["Valor_Mensal"] / 30).round(2)

    return df


# ==============================
# 7. MAIN PREPARATION PIPELINE
# ==============================

def prepare_data(source: str = "url") -> pd.DataFrame:
    """
    Full data preparation pipeline.
    """

    df = fetch_data(source)
    df = clean_data(df)
    df = translate_columns(df)
    df = standardize_values(df)
    df = convert_binary_columns(df)
    df = feature_engineering(df)
    useless_columns = ["ID_Cliente"]
    df = df.drop(columns=[col for col in useless_columns if col in df.columns])

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    
    return df
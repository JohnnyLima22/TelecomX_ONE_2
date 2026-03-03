# preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):

    # =========================
    # Detect column types
    # =========================
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # =========================
    # Remove identificadores
    # =========================
    forbidden_cols = ["ID_Cliente"]

    cat_cols = [col for col in cat_cols if col not in forbidden_cols]
    num_cols = [col for col in num_cols if col not in forbidden_cols]

    # =========================
    # ColumnTransformer
    # =========================
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore"
                ),
                cat_cols
            )
        ]
    )

    return preprocessor
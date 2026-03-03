# modeling.py

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def build_model(preprocessor, model_type="random_forest"):

    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)

        param_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20]
        }

    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000)

        param_grid = {
            "model__C": [0.1, 1, 10]
        }

    pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    return grid
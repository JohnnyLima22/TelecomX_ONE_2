# main.py

from src.config import *
from src.data_preparation import prepare_data
from src.preprocessing import build_preprocessor
from src.modeling import build_model
from src.evaluation import evaluate
from src.business_simulation import revenue_risk

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import numpy as np
import joblib


# 1. Load data
df = prepare_data()

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

if "customerID" in X.columns:
    X = X.drop("customerID", axis=1)


# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)


# 3. Preprocessor
preprocessor = build_preprocessor(X_train)


# 4. Model
model = build_model(preprocessor, MODEL_TYPE)


# ==========================
# 🔎 Cross-Validation
# ==========================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=skf,
    scoring="roc_auc"
)

print(f"Cross-Validated ROC-AUC: {cv_scores}")
print(f"Mean ROC-AUC: {np.mean(cv_scores):.4f}")
print(f"Std ROC-AUC: {np.std(cv_scores):.4f}")


# ==========================
# 🚀 Final Training
# ==========================

model.fit(X_train, y_train)


# ==========================
# 📊 Final Evaluation (TEST SET)
# ==========================

proba_test = evaluate(model, X_test, y_test)

from sklearn.metrics import precision_recall_curve
import numpy as np

precision, recall, thresholds = precision_recall_curve(y_test, proba_test)

# Remove último ponto (sklearn gera um ponto extra)
precision = precision[:-1]
recall = recall[:-1]

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best Threshold (F1 optimized): {best_threshold:.3f}")
print(f"Best F1 Score: {f1_scores[best_idx]:.3f}")
print(f"Precision at best F1: {precision[best_idx]:.3f}")
print(f"Recall at best F1: {recall[best_idx]:.3f}")
# ==========================
# 💰 Business Simulation
# ==========================

loss = revenue_risk(proba_test, AVERAGE_TICKET)
print("Estimated Annual Revenue at Risk:", loss)


# ==========================
# 💾 Save Model
# ==========================

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
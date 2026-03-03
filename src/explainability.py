# explainability.py

import shap

def run_shap(model, X_sample):

    explainer = shap.Explainer(model.named_steps["model"])
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample)
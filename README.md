![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-black?style=for-the-badge)

📊 TelecomX – End-to-End Churn Prediction & Revenue Risk Modeling
🎯 Project Overview

TelecomX is a production-oriented machine learning system designed to predict customer churn and estimate financial exposure in subscription-based models.

While most churn projects stop at classification, this system goes further by implementing Probabilistic Risk Scoring and Financial Impact Simulation, allowing stakeholders to see the "revenue at risk" in real-time.
🧠 Problem Definition

Churn prediction is treated as a binary classification task:
P(Y=1∣X)

Where:

    Y=1: The customer cancels the service.

    X: Vector of behavioral features (contract type, payment method, monthly charges).

The system prioritizes Probability Calibration and Threshold Sensitivity to align model outputs with operational retention costs.
🏗️ System Architecture & Modularity

The project follows a decoupled architecture, ensuring scalability and ease of maintenance.
Modular Breakdown:

    config.py: Centralized environment variables and model hyperparameters.

    data_loader.py: Agnostic data ingestion layer (CSV, SQL, API).

    data_preparation.py: Automated ETL and feature engineering pipeline.

    preprocessing.py: Robust scaling and encoding using ColumnTransformers.

    modeling.py: Implementation of GridSearchCV for hyperparameter optimization.

    explainability.py: Integration with SHAP for global and local interpretability.

    business_simulation.py: Financial logic for ROI and Revenue Loss calculations.

    dashboard.py: High-level UI for decision support.

📈 Model Performance & Evaluation

The system was optimized for ROC-AUC to ensure stable performance despite class imbalance.
Metric	Value
ROC-AUC	0.95
Accuracy	0.87
Precision (Churn)	0.81
Recall (Churn)	0.67

    Note on Strategy: In churn scenarios, we often prioritize Recall to capture the maximum number of potential leavers, even at the cost of some False Positives (who will still benefit from retention campaigns).

💰 Revenue Risk Modeling

Instead of a simple "Yes/No", the system calculates the Expected Revenue Loss using a probabilistic approach:
Lrevenue​=i=1∑n​(P(yi​=1∣xi​)×Ticketavg​)

This allows for a smoother financial projection that doesn't depend on a fixed binary threshold, providing a more realistic view of the Annual Recurring Revenue (ARR) at risk.
🧪 Explainability (XAI)

Using SHAP (SHapley Additive exPlanations), the system breaks down the "Black Box" of Machine Learning:

    Global: Identifies that "Month-to-Month" contracts are the primary churn drivers.

    Local: Explains to a customer service agent exactly why a specific client is at high risk.

🚀 Running the System
Bash

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the training & simulation pipeline
python main.py

# 3. Launch the Interactive Dashboard
streamlit run dashboard.py

🇧🇷 Resumo (Português)

Sistema completo de predição de churn focado em impacto financeiro. O diferencial deste projeto é a tradução de métricas técnicas (como AUC-ROC) em métricas de negócio (Receita em Risco). Utiliza arquitetura modular, validação cruzada para evitar overfitting e inteligência explicável com SHAP para dar transparência às decisões do modelo.

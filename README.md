TelecomX – End-to-End Churn Prediction & Revenue Risk Modeling System
Overview

TelecomX is a production-oriented end-to-end machine learning system for churn prediction and revenue risk estimation in subscription-based businesses.

The project demonstrates:

Modular ML pipeline design

Cross-validated model optimization

Probabilistic risk scoring

Financial impact simulation

Interactive decision-support dashboard

The system is structured for scalability, maintainability, and extensibility.

Problem Definition

Churn prediction is formulated as a binary classification problem:

P(Y=1∣X)
P(Y=1∣X)

Where:

Y=1
Y=1 indicates customer churn

X
X represents engineered behavioral and contractual features

Beyond prediction accuracy, the system focuses on:

Probability calibration

Threshold sensitivity

Revenue-weighted risk estimation

System Architecture
Data Ingestion
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
Cross-Validated Model Training
    ↓
ROC-AUC Optimization
    ↓
Probability Scoring
    ↓
Revenue Risk Simulation
    ↓
Streamlit Dashboard
Modular Structure

data_preparation.py → preprocessing & feature engineering

model_training.py → model training & evaluation

business_simulation.py → revenue impact modeling

config.py → centralized configuration

dashboard.py → interactive visualization layer

The separation of concerns enables easy refactoring and scaling.

Model Performance

ROC-AUC: 0.95

Accuracy: 0.87

Precision (churn class): 0.81

Recall (churn class): 0.67

The model demonstrates strong class separation and stable generalization under cross-validation.

ROC-AUC was selected as the primary optimization metric due to class imbalance considerations.

Revenue Risk Modeling

Revenue exposure is computed as:

Expected Revenue Loss=∑P(churni)×Average Ticket
Expected Revenue Loss=∑P(churn
i
	​

)×Average Ticket

This probabilistic approach avoids binary threshold dependency and provides smoother financial projections.

Example Output:

Estimated Annual Revenue at Risk: $515,760

Decision Threshold Engineering

The dashboard includes dynamic threshold adjustment:

Enables precision-recall tradeoff exploration

Supports business-driven decision calibration

Allows scenario simulation for retention campaigns

This aligns model outputs with operational constraints.

Explainability

The system integrates SHAP for feature-level interpretability, enabling:

Global feature importance analysis

Local explanation for individual predictions

Transparency for stakeholder communication

Dashboard Capabilities

The Streamlit dashboard provides:

ROC curve visualization

Probability distribution analysis

High-risk client ranking

Revenue exposure estimation

Individual client simulation

The UI is designed for executive and operational stakeholders.

Engineering Principles Demonstrated

Modular architecture

Reproducible training

Separation of configuration

Probabilistic modeling

Business-aligned evaluation metrics

Clean interface between ML layer and UI layer

Scalability Roadmap

Future extensions:

Model monitoring & drift detection

Batch & real-time inference API

CI/CD pipeline integration

Cloud deployment (AWS/GCP/Azure)

Automated retraining pipeline

Feature store integration

Running the Project
pip install -r requirements.txt
python main.py
streamlit run dashboard.py
What This Project Demonstrates

Strong ML fundamentals

Probabilistic modeling mindset

Revenue-aware modeling

Production-oriented code organization

Ability to connect ML outputs to business metrics

🇧🇷 (Português)
TelecomX – Sistema End-to-End de Predição de Churn e Modelagem de Risco de Receita
Visão Geral

O TelecomX é um sistema completo de Machine Learning para predição de churn e estimativa de exposição financeira em negócios baseados em assinatura.

O projeto demonstra:

Pipeline modular de ML

Otimização com validação cruzada

Score probabilístico de risco

Modelagem de impacto financeiro

Camada interativa de decisão

Estruturado para escalabilidade e manutenção em ambiente de produção.

Definição do Problema

Classificação binária:

P(Y = 1 | X)

Onde:

Y = churn

X = atributos comportamentais e contratuais

Foco não apenas em acurácia, mas em:

Calibração probabilística

Sensibilidade ao threshold

Estimativa de risco ponderada por receita

Performance

ROC-AUC: 0.95

Acurácia: 0.87

Precisão (classe churn): 0.81

Recall (classe churn): 0.67

O modelo apresenta boa generalização e forte capacidade discriminatória.

Modelagem Financeira

Receita esperada em risco:

Somatório das probabilidades individuais multiplicadas pelo ticket médio.

Isso evita dependência binária de threshold e fornece estimativa contínua de impacto.

Princípios de Engenharia

Separação de responsabilidades

Pipeline reprodutível

Métrica alinhada ao problema

Arquitetura desacoplada

Integração entre ML e camada de visualização

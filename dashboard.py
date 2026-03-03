import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from src.data_preparation import prepare_data
from src.config import TARGET_COLUMN, AVERAGE_TICKET

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="TelecomX Churn Intelligence",
    layout="wide"
)

st.title("📊 TelecomX - Churn Intelligence Dashboard")

# =============================
# LOAD DATA AND MODEL
# =============================

@st.cache_data
def load_data():
    df = prepare_data()
    return df

df = load_data()

# Carregar modelo salvo
model = joblib.load("model.pkl")

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

if "customerID" in X.columns:
    X = X.drop("customerID", axis=1)

# =============================
# PREDICTIONS
# =============================

proba = model.predict_proba(X)[:, 1]
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y, proba)

from src.business_simulation import revenue_risk
from src.config import AVERAGE_TICKET

estimated_loss = revenue_risk(proba, AVERAGE_TICKET)



st.sidebar.header("Model Controls")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

predictions = (proba >= threshold).astype(int)

st.info("""
""")

from sklearn.metrics import roc_auc_score

# =============================
# METRICS
# =============================


retention_target = predictions.sum()
potential_revenue = revenue_risk(proba[predictions == 1], AVERAGE_TICKET)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Clients", len(df))
col2.metric("Churn Rate", f"{y.mean()*100:.2f}%")
col3.metric("Clients to Target", retention_target)
col4.metric("Revenue at Risk (Targeted)", f"${potential_revenue:,.2f}")

st.divider()



# =============================
# ROC CURVE
# =============================

st.subheader("Risk Segmentation Performance")

fpr, tpr, _ = roc_curve(y, proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0,1],[0,1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

col1, col2 = st.columns([2,1])

with col1:
    st.pyplot(fig)

with col2:
    st.markdown("""
    ### 📌 Business Interpretation
    
    - High AUC indicates strong separation capacity  
    - The model effectively ranks churn risk  
    - Suitable for proactive retention strategies  
    - Low false-positive growth across threshold range  
    """)

st.divider()

# =============================
# PROBABILITY DISTRIBUTION
# =============================

st.subheader("Churn Probability Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(proba, bins=50)
ax2.set_xlabel("Predicted Probability")
ax2.set_ylabel("Number of Clients")

col1, col2 = st.columns([2,1])

with col1:
    st.pyplot(fig2)

with col2:
    st.markdown("""
    ### 🚨 Risk Concentration Insight
    
    - Distribution shows concentration of medium-risk clients  
    - High-probability tail represents revenue exposure  
    - Enables targeted intervention strategy  
    - Helps prioritize retention investment  
    """)

st.divider()
 
retention_cost = st.sidebar.number_input(
    "Retention Cost per Client ($)",
    value=50
)

expected_saved_revenue = potential_revenue * 0.60  # supondo 60% sucesso
total_cost = retention_target * retention_cost
roi = expected_saved_revenue - total_cost

st.subheader("📈 Retention Strategy Simulation")

col1, col2, col3 = st.columns(3)

col1.metric("Expected Saved Revenue", f"${expected_saved_revenue:,.2f}")
col2.metric("Total Retention Cost", f"${total_cost:,.2f}")
col3.metric("Net ROI", f"${roi:,.2f}")

st.divider()


st.markdown("""
### 🎯 Strategic Action List

The clients listed below show a higher predicted probability of churn. This list can be directly leveraged by the commercial team for targeted retention campaigns..
""")

st.success(f"""
Strategic Recommendation:

Targeting the top {retention_target} high-risk clients
could potentially protect approximately ${potential_revenue:,.2f}
in annual recurring revenue.
""")
st.subheader("🚨 Top 10 High-Risk Clients")

risk_df = X.copy()
risk_df["Churn_Probability"] = proba

top_risk = risk_df.sort_values(
    "Churn_Probability",
    ascending=False
).head(10)

st.dataframe(top_risk)

# =============================
# CLIENT SIMULATOR
# =============================

st.subheader("🔍 Individual Client Simulation")

sample_index = st.slider("Select Client Index", 0, len(X)-1, 0)

sample = X.iloc[[sample_index]]
sample_proba = model.predict_proba(sample)[0][1]

st.write("Client Data:")
st.dataframe(sample)

st.metric("Predicted Churn Probability", f"{sample_proba:.2%}")

col1, col2 = st.columns([2,1])


from sklearn.metrics import recall_score, precision_score

recall = recall_score(y, predictions)
precision = precision_score(y, predictions)

st.sidebar.markdown("### Threshold Impact")
st.sidebar.write(f"Precision: {precision:.2f}")
st.sidebar.write(f"Recall: {recall:.2f}")
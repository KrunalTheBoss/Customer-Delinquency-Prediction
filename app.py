import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Customer Delinquency Prediction",
    page_icon="📊",
    layout="wide",
)

TARGET = "default payment next month"

@st.cache_data
def load_data():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    return train_df, test_df

@st.cache_resource
def load_models():
    return {
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "Logistic Regression": joblib.load("models/logistic_model.pkl"),
    }

def get_feature_importance(model, feature_names):
    """Safely extract feature importances from Random Forest or a Logistic Regression Pipeline."""
    estimator = model[-1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        scores = np.abs(estimator.coef_[0])
    else:
        return None
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": scores})
        .sort_values("Importance", ascending=False)
    )

train_df, test_df = load_data()
models = load_models()
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# ==========================================
# SIDEBAR: Model Selection & Project Insights
# ==========================================
st.sidebar.title("🔍 Model Insights")
selected_model_name = st.sidebar.selectbox(
    "Choose a model",
    list(models.keys()),
)
model = models[selected_model_name]

# Pre-compute metrics for the selected model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

st.sidebar.divider()
st.sidebar.markdown(f"**Active Model:** {selected_model_name}")
st.sidebar.markdown(f"**Test Accuracy:** {acc * 100:.2f}%")
st.sidebar.markdown(f"**ROC-AUC Score:** {roc:.3f}")
st.sidebar.markdown(f"**Training Rows:** {len(train_df):,}")
st.sidebar.markdown(f"**Test Rows:** {len(test_df):,}")

st.sidebar.divider()
st.sidebar.markdown("🙌 **Credits**")
st.sidebar.caption("Developed by: Krunal Patel")
st.sidebar.caption("Dataset: UCI Credit Card Default")

# ==========================================
# MAIN CONTENT: 3 Clean Tabs
# ==========================================
st.title("Customer Delinquency Prediction Dashboard")
tab1, tab2, tab3 = st.tabs([
    "📊 Overview Dashboard",
    "👥 Customer Prediction",
    "📈 Model Performance & Importance",
])

# --- TAB 1: Professional Dashboard ---
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers (Train)", f"{len(train_df):,}")
    k2.metric("Overall Default Rate", f"{train_df[TARGET].mean() * 100:.2f}%")
    k3.metric("Avg Credit Limit", f"{train_df['LIMIT_BAL'].mean():,.0f}")
    k4.metric("Selected Model Accuracy", f"{acc * 100:.2f}%")

    st.divider()

    # Blue-gradient bar chart with percentage labels (matching reference style)
    pay_summary = (
        train_df.groupby("PAY_0")[TARGET]
        .agg(Customers="count", Default_Rate="mean")
        .reset_index()
    )
    pay_summary["Default_Rate_Pct"] = pay_summary["Default_Rate"] * 100
    pay_summary["PAY_0"] = pay_summary["PAY_0"].astype(str)

    fig_bar = px.bar(
        pay_summary,
        x="PAY_0",
        y="Default_Rate_Pct",
        color="Default_Rate_Pct",
        color_continuous_scale="Blues",
        text=pay_summary["Default_Rate_Pct"].apply(lambda x: f"{x:.1f}%"),
        template="plotly_dark",
        title="Default Rate (%) by Recent Repayment Status (PAY_0)",
        labels={"PAY_0": "Repayment Status (PAY_0)", "Default_Rate_Pct": "Default Rate (%)"},
    )
    fig_bar.update_traces(textposition="inside")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Customer Records Preview")
    st.dataframe(train_df.head(15), use_container_width=True)

# --- TAB 2: Customer Prediction ---
with tab2:
    st.subheader(f"Predict Default Risk ({selected_model_name})")
    st.caption("Enter customer financial details below. Unspecified historical columns default to dataset medians.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        limit_bal = c1.number_input("Credit Limit (LIMIT_BAL)", 10000, 1000000, 150000, step=10000)
        age = c2.slider("Age", 21, 75, 32)
        education = c3.selectbox(
            "Education",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Grad School", 2: "University", 3: "High School", 4: "Other"}[x],
            index=1,
        )

        c4, c5, c6 = st.columns(3)
        pay_0 = c4.selectbox("Last Month Status (PAY_0)", [-2, -1, 0, 1, 2, 3, 4], index=2)
        pay_2 = c5.selectbox("2 Months Ago Status (PAY_2)", [-2, -1, 0, 1, 2, 3, 4], index=2)
        bill_amt1 = c6.number_input("Latest Bill Amount (BILL_AMT1)", -50000, 1000000, 50000, step=5000)

        pay_amt1 = st.number_input("Latest Payment Amount (PAY_AMT1)", 0, 500000, 5000, step=1000)

        submitted = st.form_submit_button("Predict Default Risk", type="primary")

    if submitted:
        input_row = train_df.drop(columns=[TARGET]).median().to_frame().T
        input_row["LIMIT_BAL"] = limit_bal
        input_row["AGE"] = age
        input_row["EDUCATION"] = education
        input_row["PAY_0"] = pay_0
        input_row["PAY_2"] = pay_2
        input_row["BILL_AMT1"] = bill_amt1
        input_row["PAY_AMT1"] = pay_amt1

        prob = model.predict_proba(input_row)[0][1]
        pred = model.predict(input_row)[0]

        st.divider()
        r1, r2 = st.columns(2)
        r1.metric("Predicted Default Probability", f"{prob * 100:.1f}%")
        if pred == 1:
            r2.error("High Risk: Likely to Default")
        else:
            r2.success("Low Risk: Likely to Pay on Time")

# --- TAB 3: Model Performance & Feature Importance ---
with tab3:
    st.subheader(f"Evaluation Metrics — {selected_model_name}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc * 100:.2f}%")
    m2.metric("Precision", f"{precision_score(y_test, y_pred) * 100:.2f}%")
    m3.metric("Recall", f"{recall_score(y_test, y_pred) * 100:.2f}%")
    m4.metric("F1-Score", f"{f1_score(y_test, y_pred) * 100:.2f}%")
    m5.metric("ROC-AUC", f"{roc:.3f}")

    st.divider()
    col_left, col_right = st.columns([3, 2])

    with col_left:
        imp_df = get_feature_importance(model, X_test.columns).head(10)
        fig_imp = px.bar(
            imp_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            template="plotly_dark",
            title=f"Top 10 Most Important Features ({selected_model_name})",
        )
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_right:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Pred: No Default", "Pred: Default"],
            y=["Actual: No Default", "Actual: Default"],
            template="plotly_dark",
            title="Confusion Matrix (Test Set)",
        )
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)
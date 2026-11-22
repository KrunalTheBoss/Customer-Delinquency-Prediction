import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Customer Delinquency Prediction",
    page_icon="📊",
    layout="wide"
)

# Dashboard title
st.title("Customer Delinquency Prediction")
st.subheader("AI & Data Science Machine Learning Project")

st.markdown("""
Welcome to the Customer Delinquency Prediction Dashboard.

This application analyses customer financial data
and demonstrates Machine Learning-based predictions.
""")

# Load dataset
df = pd.read_csv("data/processed/train.csv")

# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("Training Customers", f"{len(df):,}")
col2.metric("Features", "23")
col3.metric(
    "Default Rate",
    f"{df['default payment next month'].mean() * 100:.2f}%"
)

st.divider()

# Dataset preview
st.header("Customer Dataset")
st.dataframe(df.head(10), use_container_width=True)

# Default distribution
st.header("Customer Default Distribution")

distribution = (
    df["default payment next month"]
    .value_counts()
    .rename(index={0: "No Default", 1: "Default"})
)

st.bar_chart(distribution)

st.success("Dashboard is running successfully!")
# 📊 Customer Delinquency Prediction

An end-to-end Machine Learning project and interactive Streamlit BI dashboard that predicts whether a credit card customer will default on their next monthly payment.

> **Disclaimer:** This is an educational and portfolio project built on a historical public dataset. It is not a validated system for real-world lending or credit decisions.

---

## 🚀 Features

- **Interactive BI Dashboard:** Dark-mode Streamlit interface powered by Plotly visualizations.
- **Model Selection:** Switch seamlessly between **Logistic Regression** and **Random Forest** models in the sidebar.
- **Live Risk Simulator:** Input customer financial details (credit limit, age, education, repayment history, bill & payment amounts) to generate real-time default probability estimates.
- **Model Performance & Explainability:** View live test-set metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC), interactive Confusion Matrices, and Top-10 Feature Importance charts.

---

## 📂 Dataset

- **Name:** Default of Credit Card Clients
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Size:** 30,000 records, 23 predictive features (after removing `ID`), 0 missing values, 0 duplicates.
- **Target Variable:** `default payment next month`

| Target | Meaning | Count | Percentage |
| :--- | :--- | ---: | ---: |
| `0` | No Default | 23,364 | 77.88% |
| `1` | Default | 6,636 | 22.12% |

Because the classes are imbalanced, the dataset was split using a **stratified 80/20 train-test split** (24,000 training rows / 6,000 testing rows) and evaluated beyond raw accuracy using Precision, Recall, F1-Score, and ROC-AUC.

---

## 🤖 Models & Baseline Results

| Model | Configuration | Test Accuracy |
| :--- | :--- | ---: |
| **Logistic Regression** | `StandardScaler` + `LogisticRegression(max_iter=1000)` | 80.77% |
| **Random Forest** | `RandomForestClassifier(n_estimators=100, class_weight="balanced")` | 80.02% |

*Full classification metrics (Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix) are computed and displayed dynamically inside the dashboard.*

---

## 🗂️ Project Structure

```text
Customer-Delinquency-Prediction/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   │   └── default of credit card clients.xls
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── logistic_model.pkl
│   └── random_forest_model.pkl
├── reports/
│   ├── eda_analysis.png
│   ├── logistic_regression_confusion_matrix.png
│   └── random_forest_confusion_matrix.png
├── src/
│   ├── load_data.py
│   ├── understand_data.py
│   ├── eda.py
│   ├── prepare_data.py
│   ├── train_model.py
│   └── evaluate_model.py
├── app.py
├── requirements.txt
└── README.md
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

# Load testing data
test = pd.read_csv("data/processed/test.csv")

target = "default payment next month"

X_test = test.drop(columns=[target])
y_test = test[target]

# Load trained models
models = {
    "Logistic Regression": joblib.load(
        "models/logistic_model.pkl"
    ),
    "Random Forest": joblib.load(
        "models/random_forest_model.pkl"
    )
}

# Evaluate each model
for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:")
    print(round(accuracy_score(y_test, predictions), 4))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nROC-AUC Score:")
    print(round(roc_auc_score(y_test, probabilities), 4))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"]
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        f"reports/{name.replace(' ', '_').lower()}_confusion_matrix.png",
        dpi=300
    )

    plt.show()

print("\nMODEL EVALUATION COMPLETED!")
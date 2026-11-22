import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load training and testing data
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

target = "default payment next month"

X_train = train.drop(columns=[target])
y_train = train[target]

X_test = test.drop(columns=[target])
y_test = test[target]

# Model 1: Logistic Regression
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test, logistic_predictions
)

# Model 2: Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

# Display results
print("\nMODEL TRAINING COMPLETED")

print("\nLogistic Regression Accuracy:")
print(round(logistic_accuracy * 100, 2), "%")

print("\nRandom Forest Accuracy:")
print(round(rf_accuracy * 100, 2), "%")

# Save trained models
joblib.dump(logistic_model, "models/logistic_model.pkl")
joblib.dump(random_forest, "models/random_forest_model.pkl")

print("\nModels saved successfully!")
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load dataset
file_path = "data/raw/default of credit card clients.xls"
df = pd.read_excel(file_path, header=1)

# Remove customer ID
df = df.drop(columns=["ID"])

# Separate features and target
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"]

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create output folder
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

# Save training and testing datasets
train_data = X_train.copy()
train_data["default payment next month"] = y_train

test_data = X_test.copy()
test_data["default payment next month"] = y_test

train_data.to_csv(output_folder / "train.csv", index=False)
test_data.to_csv(output_folder / "test.csv", index=False)

# Display results
print("\nDATA PREPROCESSING COMPLETED")

print("\nTraining dataset:", X_train.shape)
print("Testing dataset:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True) * 100)

print("\nFiles saved successfully!")
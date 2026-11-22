import pandas as pd

# Load dataset
file_path = "data/raw/default of credit card clients.xls"
df = pd.read_excel(file_path, header=1)

# 1. Dataset dimensions
print("\n--- DATASET SHAPE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sum())

# 3. Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())

# 4. Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 5. Target variable distribution
print("\n--- DEFAULT DISTRIBUTION ---")
print(df["default payment next month"].value_counts())

# 6. Target variable percentages
print("\n--- DEFAULT PERCENTAGES ---")
print(
    df["default payment next month"]
    .value_counts(normalize=True) * 100
)
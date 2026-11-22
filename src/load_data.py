import pandas as pd

# Load the customer dataset
file_path = "data/raw/default of credit card clients.xls"

df = pd.read_excel(file_path, header=1)

# Display the first five records
print("\nFIRST FIVE CUSTOMER RECORDS")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Display dataset information
print("\nDATASET INFORMATION")
df.info()
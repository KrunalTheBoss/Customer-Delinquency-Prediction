import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/raw/default of credit card clients.xls"
df = pd.read_excel(file_path, header=1)

# Set visualisation style
sns.set_theme(style="whitegrid")

# Create a figure with four charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Customer default distribution
sns.countplot(
    data=df,
    x="default payment next month",
    hue="default payment next month",
    legend=False,
    ax=axes[0, 0]
)
axes[0, 0].set_title("Customer Default Distribution")
axes[0, 0].set_xlabel("Default Next Month (0 = No, 1 = Yes)")

# 2. Customer age distribution
sns.histplot(
    data=df,
    x="AGE",
    bins=30,
    kde=True,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Customer Age Distribution")

# 3. Credit limit distribution
sns.histplot(
    data=df,
    x="LIMIT_BAL",
    bins=30,
    kde=True,
    ax=axes[1, 0]
)
axes[1, 0].set_title("Customer Credit Limit Distribution")

# 4. Default status by repayment status
sns.countplot(
    data=df,
    x="PAY_0",
    hue="default payment next month",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Default by Recent Repayment Status")
axes[1, 1].set_xlabel("Most Recent Repayment Status (PAY_0)")

# Improve layout
plt.tight_layout()

# Save chart
plt.savefig("reports/eda_analysis.png", dpi=300)

# Display charts
plt.show()

print("EDA analysis completed successfully!")
print("Chart saved in reports/eda_analysis.png")
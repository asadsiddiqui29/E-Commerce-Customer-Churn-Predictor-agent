import pandas as pd
import numpy as np

np.random.seed(42)

rows = 10000

data = {
    "LoginFreq": np.random.randint(1, 31, rows),
    "Purchases": np.random.randint(0, 21, rows),
    "AmountSpent": np.random.randint(10, 5000, rows),
    "Tickets": np.random.randint(0, 11, rows),
    "DaysInactive": np.random.randint(1, 181, rows),
    "MembershipMonths": np.random.randint(1, 61, rows)
}

df = pd.DataFrame(data)

df["Churn"] = (
    ((df["DaysInactive"] > 90) & (df["Purchases"] < 5))
    |
    ((df["LoginFreq"] < 5) & (df["AmountSpent"] < 500))
).astype(int)

df.to_csv("ecommerce_churn.csv", index=False)

print("Dataset Generated Successfully!")
print(f"Total Rows: {len(df)}")
print(df.head())
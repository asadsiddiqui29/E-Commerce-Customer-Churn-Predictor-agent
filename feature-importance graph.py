#this part feature importance graph tells us the behavior among customer that influence churn the most.(using the dataset created)

import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("churn_model.pkl")

df = pd.read_csv("ecommerce_churn.csv")

X = df.drop("Churn", axis=1)

importance = model.feature_importances_

  

plt.figure(figsize=(8,5))
plt.barh(X.columns, importance)
plt.xlabel("Importance Score")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
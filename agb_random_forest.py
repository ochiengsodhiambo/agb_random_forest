
---

##  Clean Python Script 

Save as `agb_random_forest.py`:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/2015_sampled_indices.csv")

# Preview data
print(df.head())

# Data inspection
print(df.info())
print(df.describe())

# Clean data
df = df.dropna()

# Features and target
X = df[["evi_2015me", "ndvi_2015m"]]
y = df["agb_mean"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

print("First 5 predictions:", y_pred[:5])

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")

plt.xlabel("Actual AGB")
plt.ylabel("Predicted AGB")
plt.title("Random Forest: AGB Prediction")

plt.show()

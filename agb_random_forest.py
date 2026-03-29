import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# loading my data
df = pd.read_csv("D:\\other\\4me\\mwache\\py_est\\2015_sampled_indices.csv")

# Inspect first rows
print(df.head())

#Check for missing values, wrong data types, or outliers.
print(df.info())
print(df.describe())

# Drop rows with missing values
df = df.dropna()

# Split Features (X) and Target (y)
X = df[["evi_2015me", "ndvi_2015m"]]   # predictors
y = df["agb_mean"]              # target

#Split into training and testing sets (helps to evaluate performance)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Training the Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make Predictions
# Predict weights for test set
y_pred = rf.predict(X_test)

print("First 5 predictions:", y_pred[:5])

#Evaluate Model Performance
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

#Visualize Results
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual agb")
plt.ylabel("Predicted agb")
plt.title("Random Forest: AGB Prediction")
plt.show()


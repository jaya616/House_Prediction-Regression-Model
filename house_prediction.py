# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load dataset
data = pd.read_csv("train.csv")

# Show first 5 rows
print(data.head())

# Check shape of dataset
print("Dataset Shape:", data.shape)

# Check missing values
print(data.isnull().sum().head(20))

# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Fill missing numeric values with mean
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing categorical values with mode (most frequent value)
for col in categorical_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])

# Drop columns with too many missing values (optional for beginners)
data = data.dropna(axis=1, thresh=0.8*len(data))

# Convert categorical columns to numbers (One-Hot Encoding)
data = pd.get_dummies(data, drop_first=True)

# Target = SalePrice
y = data["SalePrice"]

# Features = all other columns except SalePrice
X = data.drop("SalePrice", axis=1)

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred[:5])
print("Actual Prices:", list(y_test[:5]))

# Mean Squared Error (MSE) - lower is better
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R² Score - closer to 1 is better
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Plot Actual vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

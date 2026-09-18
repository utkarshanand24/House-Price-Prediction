#python
# ============================================================
# HOUSE PRICE PREDICTION PROJECT
# ============================================================
# Dataset: Kaggle House Prices - Ames Housing Dataset
# Models: Linear Regression & Random Forest
# ============================================================


# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -----------------------------
# 2. LOAD DATASET
# -----------------------------

df = pd.read_csv("dataset/train.csv")

print("=" * 60)
print("HOUSE PRICE PREDICTION PROJECT")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# -----------------------------
# 3. MISSING VALUE ANALYSIS
# -----------------------------

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_data = pd.DataFrame({
    "Missing Values": missing_values,
    "Percentage": missing_percentage
})

missing_data = missing_data[
    missing_data["Missing Values"] > 0
].sort_values(
    by="Missing Values",
    ascending=False
)

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

print(missing_data)


# -----------------------------
# 4. DATA CLEANING
# -----------------------------

# Missing values that represent
# absence of a particular feature
none_columns = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "MasVnrType",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2"
]

for col in none_columns:
    df[col] = df[col].fillna("None")


# Numerical missing values
df["LotFrontage"] = df["LotFrontage"].fillna(
    df["LotFrontage"].median()
)

df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)


# Categorical missing value
df["Electrical"] = df["Electrical"].fillna(
    df["Electrical"].mode()[0]
)


print("\nMissing Values After Cleaning:")
print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .head(20)
)


# -----------------------------
# 5. EXPLORATORY DATA ANALYSIS
# -----------------------------

# Sale Price Distribution
plt.figure(figsize=(10, 5))

sns.histplot(
    df["SalePrice"],
    kde=True
)

plt.title("Distribution of House Prices")
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.show()


# Overall Quality vs Sale Price
plt.figure(figsize=(10, 5))

sns.boxplot(
    x="OverallQual",
    y="SalePrice",
    data=df
)

plt.title("Overall Quality vs Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.tight_layout()
plt.show()


# -----------------------------
# 6. CORRELATION ANALYSIS
# -----------------------------

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = (
    numeric_df
    .corr()["SalePrice"]
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("CORRELATION WITH SALEPRICE")
print("=" * 60)

print(correlation.head(15))


# Top 10 Correlated Features
plt.figure(figsize=(10, 6))

sns.barplot(
    x=correlation.head(10).values,
    y=correlation.head(10).index
)

plt.title("Top 10 Features Correlated with Sale Price")
plt.xlabel("Correlation")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# -----------------------------
# 7. FEATURE SELECTION
# -----------------------------

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "YearBuilt"
]

X = df[features]
y = df["SalePrice"]

print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

print("\nSelected Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# -----------------------------
# 8. TRAIN-TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================================
# 9. LINEAR REGRESSION MODEL
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

print("\nLinear Regression Training Completed!")


# Predictions
linear_predictions = linear_model.predict(X_test)


# Evaluation
mae = mean_absolute_error(
    y_test,
    linear_predictions
)

mse = mean_squared_error(
    y_test,
    linear_predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    linear_predictions
)


print("\nLinear Regression Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

print("\nRandom Forest Training Completed!")


# Predictions
rf_predictions = rf_model.predict(X_test)


print("\nFirst 10 Random Forest Predictions:")
print(rf_predictions[:10])

print("\nFirst 10 Actual Prices:")
print(y_test.values[:10])


# -----------------------------
# 11. RANDOM FOREST EVALUATION
# -----------------------------

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_mse = mean_squared_error(
    y_test,
    rf_predictions
)

rf_rmse = rf_mse ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


print("\nRandom Forest Model Evaluation:")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


# -----------------------------
# 12. ACTUAL VS PREDICTED
# -----------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.6
)

plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()
plt.show()


# -----------------------------
# 13. FEATURE IMPORTANCE
# -----------------------------

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=feature_importance.values,
    y=feature_importance.index
)

plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()
plt.show()


# -----------------------------
# 14. MODEL COMPARISON
# -----------------------------

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        mae,
        rf_mae
    ],
    "RMSE": [
        rmse,
        rf_rmse
    ],
    "R2 Score": [
        r2,
        rf_r2
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison)


# ============================================================
# 15. NEW HOUSE PRICE PREDICTION
# ============================================================

new_house = pd.DataFrame({
    "OverallQual": [7],
    "GrLivArea": [1800],
    "GarageCars": [2],
    "GarageArea": [500],
    "TotalBsmtSF": [1000],
    "1stFlrSF": [1000],
    "FullBath": [2],
    "YearBuilt": [2005]
})


predicted_price = rf_model.predict(
    new_house
)


print("\n" + "=" * 60)
print("NEW HOUSE PRICE PREDICTION")
print("=" * 60)

print("\nNew House Details:")
print(new_house)

print("\nPredicted House Price:")
print(f"${predicted_price[0]:,.2f}")


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 60)


# ==========================================
# STEP 22: EXPORT DATA FOR POWER BI
# ==========================================

# 1. Main house price data
powerbi_data = df[features + ["SalePrice"]].copy()

powerbi_data.to_csv(
    "house_price_powerbi.csv",
    index=False
)

# 2. Actual vs Predicted prices
test_results = X_test.copy()

test_results["ActualPrice"] = y_test.values
test_results["PredictedPrice"] = rf_predictions

test_results.to_csv(
    "model_predictions.csv",
    index=False
)

# 3. Model performance
model_metrics = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae, rf_mae],
    "RMSE": [rmse, rf_rmse],
    "R2 Score": [r2, rf_r2]
})

model_metrics.to_csv(
    "model_metrics.csv",
    index=False
)

print("\n==========================================")
print("POWER BI FILES CREATED SUCCESSFULLY!")
print("==========================================")
print("1. house_price_powerbi.csv")
print("2. model_predictions.csv")
print("3. model_metrics.csv")

# ==========================================
# EXPORT FEATURE IMPORTANCE FOR POWER BI
# ==========================================

feature_importance_df = pd.DataFrame({
    "Feature": feature_importance.index,
    "Importance": feature_importance.values
})

feature_importance_df.to_csv("feature_importance.csv", index=False)

print("\nFeature Importance file created successfully!")
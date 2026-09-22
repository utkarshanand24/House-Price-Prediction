# House Price Prediction

A Machine Learning web application that predicts house prices using Random Forest Regression, with an interactive Streamlit interface and Power BI dashboard.

## 🚀 Live Demo

[Try House Price AI](https://house-price-prediction-b4kmj6hwlzgzyqjjaqwkve.streamlit.app/)

## ✨ Features

- House price prediction using Machine Learning
- Random Forest Regression
- Interactive Streamlit web application
- User-friendly house property inputs
- Prediction history
- Feature importance visualization
- Model performance metrics
- Actual vs Predicted price visualization
- Power BI dashboard

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Power BI

## 🤖 Machine Learning Models

- Linear Regression
- Random Forest Regressor

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 24,932.67 | 39,558.88 | 0.796 |
| Random Forest | 19,285.42 | 29,992.91 | 0.883 |

## 🔍 Selected Features

The Random Forest model uses the following house features:

- OverallQual
- GrLivArea
- GarageCars
- GarageArea
- TotalBsmtSF
- 1stFlrSF
- FullBath
- YearBuilt

## 📈 Feature Importance

The most important features identified by the Random Forest model include:

- OverallQual
- GrLivArea
- TotalBsmtSF
- 1stFlrSF
- YearBuilt
- GarageArea
- GarageCars
- FullBath

## 📸 Project Screenshots

### 🏠 Streamlit Web Application

![Streamlit App](screenshots/streamlit_app.png)

## 📊 Power BI Dashboard

The Power BI dashboard includes:

- Total Houses
- Average Sale Price
- Price Distribution
- Overall Quality vs Sale Price
- Model Performance
- Actual vs Predicted House Prices
- Feature Importance

### Dashboard Preview

![Power BI Dashboard](screenshots/powerbi_dashboard.png)

## 🌐 Streamlit Web Application

This project also includes an interactive web application built using Streamlit.

### Features

- House price prediction
- Random Forest Regression
- User-friendly house property inputs
- Prediction history
- Feature importance visualization
- Model performance metrics
- Actual vs Predicted price visualization

### Run the Application

Install the required libraries:

```bash
pip install streamlit pandas scikit-learn

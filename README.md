# House Price Prediction

A Machine Learning project that predicts house prices using Python and Random Forest Regression.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Power BI

## Machine Learning Models

- Linear Regression
- Random Forest Regressor

## Selected Features

- OverallQual
- GrLivArea
- GarageCars
- GarageArea
- TotalBsmtSF
- 1stFlrSF
- FullBath
- YearBuilt

## Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 24,932.67 | 39,558.88 | 0.796 |
| Random Forest | 19,285.42 | 29,992.91 | 0.883 |

## Power BI Dashboard

The dashboard includes:

- Total Houses
- Average Sale Price
- Price Distribution
- Overall Quality vs Sale Price
- Model Performance
- Actual vs Predicted House Prices
- Feature Importance

## Streamlit Web Application

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
## Project Structure

```text
House-Price-Prediction/
├── house_price.py
├── house_price_powerbi.csv
├── model_predictions.csv
├── model_metrics.csv
├── feature_importance.csv
└── README.md
## Live Demo

🚀 [Try House Price AI](https://house-price-prediction-b4kmj6hwlzgzyqjjaqwkve.streamlit.app/)

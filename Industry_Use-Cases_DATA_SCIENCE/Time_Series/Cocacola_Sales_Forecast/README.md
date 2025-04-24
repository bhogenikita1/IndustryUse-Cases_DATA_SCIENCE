## 🥤 Coca-Cola Sales Forecasting (Time Series Analysis)
This project analyzes Coca-Cola’s quarterly sales data from 1986 to 1996 and uses various time series forecasting techniques to predict sales for the next two years (1997–1998). The notebook explores seasonality, trend, and cyclic behavior, and applies both classical and machine learning-based forecasting methods.

## 📘 Project Notebook
File: CocaCola_Sales_Forecast.ipynb

#Contents:

1. Data exploration and visualization
2. Dummy variable creation for seasonality
3. Model training and testing
4. RMSE evaluation
5. Sales forecasting for future quarters

## 🧾 Dataset Details
Product: Coca-Cola

Time Frame: Q1 1986 to Q4 1996 (Quarterly data)

Target: Sales volume

## ⚙️ Techniques Used
1. Data Cleaning & Transformation
2. Feature Engineering with dummy variables (Q1, Q2, Q3, Q4)

#Classical Forecasting Models:
1. Linear Trend
2. Exponential Trend
3. Additive Seasonality
4. Multiplicative Seasonality
5. Additive Seasonality with Trend
6. RMSE for model comparison
7. Final model selection for future forecasting

## 📊 Evaluation Metric
RMSE (Root Mean Squared Error) used to assess model performance on the test dataset.

## 📈 Forecast Output
Provides predicted Coca-Cola sales for Q1 1997 to Q4 1998.

## 🛠️ Requirements
Python 3.8+

## Libraries:

pandas  
numpy  
matplotlib  
seaborn  
statsmodels  

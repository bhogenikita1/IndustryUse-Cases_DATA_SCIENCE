# 🔆 Solar Power Generation Forecasting (Time Series Analysis)

This project analyzes and forecasts solar power generation using time series techniques. The goal is to understand generation patterns and predict future solar output based on historical data.

## 📘 Notebook

- **File**: `SolarPower.ipynb`  
- **Purpose**: Time series forecasting of solar power generation  
- **Includes**:
  - Data preprocessing and visualization  
  - Trend and seasonality detection  
  - Feature engineering with dummy variables  
  - Model training and performance comparison using RMSE  
  - Forecasting future solar generation

## 🔢 Dataset Information

- **Metric**: Solar power generation (kWh or similar unit)  
- **Time Frame**: Hourly/Daily/Monthly (based on actual dataset)  
- **Variables**: Timestamp, Power generated

## ⚙️ Techniques Used

- Trend & Seasonality Analysis  
- Time Series Decomposition  
- Dummy Variable Creation  
- Forecasting Models:
  - Linear Model
  - Exponential Model
  - Additive and Multiplicative Seasonality Models
  - Seasonality with Trend Models
- Model Evaluation using **RMSE**

## 🔮 Model Output

- Selection of the best model with lowest RMSE  
- Forecast for future solar power generation values

## 🛠️ Requirements

- Python 3.8+
  
- Libraries:
  
  pandas  
  numpy  
  matplotlib  
  seaborn  
  statsmodels  

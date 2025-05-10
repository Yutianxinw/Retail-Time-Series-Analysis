# 📦 Walmart M5 Forecasting: Time Series Prediction with LSTM & GRU

This project tackles the M5 Forecasting Accuracy Challenge by predicting unit sales of Walmart retail goods. We applied deep learning models (LSTM, GRU) and a baseline LightGBM model to capture complex temporal patterns and improve predictive performance.

## 🎯 Objective

Forecast daily sales for thousands of Walmart items across stores in CA, TX, and WI using the official [M5 Forecasting dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview). The final model was submitted to Kaggle and scored using the **WRMSSE** metric.

## 🧠 Models Explored

| Model            | Optimizer | Epochs | Dropout | Hidden Layers | Public Score | Private Score |
|------------------|-----------|--------|---------|----------------|---------------|----------------|
| LSTM (Adam)      | Adam      | 20     | 0.2     | 3              | 0.85804       | 0.82101        |
| LSTM (AdamW)     | AdamW     | 50     | 0.3     | 2              | 1.34016       | 1.23617        |
| GRU              | Adam      | 30     | 0.2     | 2              | 1.32166       | 1.22771        |
| LightGBM         | -         | -      | -       | -              | Not Submitted | Not Submitted  |

## 🧱 Feature Engineering

Features used to enhance signal quality and model learning:

- **Lag features** (e.g., demand from 7/28 days ago)
- **Rolling statistics** (means and stds over 7–30 day windows)
- **Calendar-based**: day of week, holidays, SNAP flags
- **Price dynamics**: price normalization, change rate
- **Hierarchical sales ratios**: store/category/department-level aggregates

## ⚙️ Technical Stack

- Python (TensorFlow/Keras, pandas, numpy, LightGBM)
- Notebooks: `LSTM(Adam).ipynb`, `LSTM(AdamW).ipynb`, `GRU.ipynb`
- Evaluation metric: WRMSSE (Weighted Root Mean Squared Scaled Error)



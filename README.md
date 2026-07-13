# End-to-End Sales Forecasting & Demand Intelligence System

## Project Overview

This project presents an end-to-end Sales Forecasting and Demand Intelligence System developed using Python. The project analyzes retail sales data, performs forecasting using multiple machine learning and statistical models, detects anomalies, segments products based on demand, and provides an interactive Streamlit dashboard for business decision-making.

---

## Objectives

- Perform exploratory data analysis on retail sales data.
- Forecast future sales using multiple forecasting techniques.
- Compare forecasting models using evaluation metrics.
- Detect unusual sales patterns using anomaly detection.
- Segment products using K-Means clustering.
- Develop an interactive Streamlit dashboard.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- XGBoost
- Streamlit

---

## Models Used

### Model 1
- SARIMA

### Model 2
- Prophet (Installation successful but runtime error occurred on Windows)

### Model 3
- XGBoost

---

## Features

- Sales Overview Dashboard
- Monthly Sales Trend
- Category & Region Analysis
- Sales Forecasting
- Anomaly Detection
- Product Demand Segmentation
- Interactive Streamlit Dashboard

---

## Evaluation Metrics

| Model | MAE | RMSE |
|------|------:|------:|
| SARIMA | 13455.42 | 15938.99 |
| XGBoost | 8826.43 | 13342.42 |

XGBoost achieved the lowest prediction error and was selected as the best-performing forecasting model.

---

## Project Structure

```
SalesForecasting_Simran/
│
├── app.py
├── SalesForecasting.ipynb
├── train.csv
├── vgsales.csv
├── requirements.txt
├── README.md
├── charts/
└── screenshots/
```

---

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Author

**Simran**

B.Tech Computer Science & Engineering

ABES Engineering College



## Live Demo

**Streamlit App:**  
https://salesforecastingsimran-zmt7t7wrlokzudpzvjw95h.streamlit.app/

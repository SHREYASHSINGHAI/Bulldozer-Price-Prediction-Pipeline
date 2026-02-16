# Bulldozer Price Prediction Pipeline

An end-to-end Machine Learning pipeline that predicts bulldozer auction sale prices using historical sales data, feature engineering, and a Random Forest regression model.

This project demonstrates a complete ML workflow including data preprocessing, feature engineering, model training, evaluation, and prediction generation.

---

## Problem Statement

Predict the future sale price of bulldozers using historical auction data and machine characteristics.

---

## Dataset

Dataset used: **Bluebook for Bulldozers** competition dataset.

The dataset contains historical auction results and equipment attributes.

Main datasets:
- **Train.csv** – Training data (up to end of 2011)
- **Valid.csv** – Validation data (Jan–Apr 2012)
- **Test.csv** – Test data (May–Nov 2012)

Key fields include:
- `SalesID` – Unique sale identifier
- `MachineID` – Machine identifier
- `SalePrice` – Auction price (training only)
- `saledate` – Date of sale

Dataset download:
https://www.kaggle.com/competitions/bluebook-for-bulldozers/data

*(Dataset files are not included due to size limits.)*

---

## Project Pipeline

The pipeline includes:

1. Data loading
2. Data exploration
3. Data cleaning
4. Feature engineering from sale dates
5. Missing value handling
6. Categorical encoding
7. Train–validation split
8. Random Forest model training
9. Hyperparameter tuning
10. Model evaluation
11. Test data preprocessing
12. Prediction generation
13. Feature importance analysis

---

## Evaluation Metric

Model performance is measured using:

**RMSLE (Root Mean Squared Log Error)**

This metric penalizes large prediction errors while handling skewed price distributions effectively.

---

## Model Used

Random Forest Regressor was used due to:
- Strong performance on tabular data
- Robustness to feature scaling
- Ability to model nonlinear relationships

---

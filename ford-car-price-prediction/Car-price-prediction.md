# 🚗 Ford Car Price Prediction

A machine learning project built for a competition, focused on exploratory data analysis (EDA) and price prediction for used Ford cars using regression techniques.

## 📌 Project Overview

This project analyzes a dataset of used Ford car listings to understand what drives resale price, then builds a regression model to predict a car's price based on its specifications (model, year, mileage, engine size, fuel type, etc.).

**Workflow:**
1. Exploratory Data Analysis (EDA) on the Ford car listings dataset
2. Feature engineering — comparing One-Hot Encoding vs. Label Encoding for categorical variables
3. Feature scaling with StandardScaler
4. Linear Regression model training and evaluation (R² and Adjusted R²)

## 📊 Dataset

The dataset (`ford.csv`) contains **17,966 used car listings** (after removing duplicates) with the following features:

| Column | Description | Type |
|---|---|---|
| `model` | Ford car model (e.g. Fiesta, Focus) | Categorical |
| `year` | Year of manufacture | Numeric |
| `price` | Sale price (target variable) | Numeric |
| `transmission` | Transmission type (Manual, Automatic, etc.) | Categorical |
| `mileage` | Distance driven (miles) | Numeric |
| `fuelType` | Fuel type (Petrol, Diesel, etc.) | Categorical |
| `tax` | Road tax | Numeric |
| `mpg` | Miles per gallon (fuel efficiency) | Numeric |
| `engineSize` | Engine size (litres) | Numeric |

No missing values were found in the dataset.

## 🔍 Exploratory Data Analysis

The EDA covered:
- Distribution of `price` (histogram with KDE) and outlier detection (boxplot)
- Correlation heatmap across all numeric features
- Price trends over `year` (boxplot)
- Relationship between `mileage` and `price` (scatterplot)
- Price variation across `engineSize`, `transmission`, `fuelType`, `model`, `tax`, and `mpg` (boxplots)

**Key findings:**
- Newer cars (`year`) and lower `mileage` are generally associated with higher prices.
- `engineSize` and `model` show a clear influence on price — larger engines and certain models command a premium.
- Categorical features like `transmission` and `fuelType` show visible price differences across their groups.

## 🤖 Model & Results

**Preprocessing approaches compared:**
- **One-Hot Encoding** for `model`, `transmission`, `fuelType` + StandardScaler on numeric columns
- **Label Encoding** for the same categorical columns + StandardScaler on all columns

**Model:** Linear Regression (train-test split, `test_size=0.33`, `random_state=42`)

| Encoding Strategy | R² Score | Adjusted R² |
|---|---|---|
| **One-Hot Encoding** | **0.8397** | **0.8387** |
| Label Encoding | 0.7310 | — |

**One-Hot Encoding significantly outperformed Label Encoding** for this dataset — likely because Label Encoding imposes an artificial ordinal relationship on categorical variables (like `model`) that don't have a natural order, misleading the linear regression model.

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — EDA visualizations
- **Scikit-learn** — preprocessing (encoding, scaling), Linear Regression, evaluation metrics

## 📁 Project Structure

```
ford-car-price-prediction/
│
├── ford.csv                       # Dataset
├── Ford_Price_Prediction.ipynb    # EDA, preprocessing & model training notebook
└── Car-price-prediction.md
```

## 🚀 Future Improvements

- Try non-linear models (Random Forest, XGBoost, Gradient Boosting) to capture non-linear price relationships
- Perform hyperparameter tuning and cross-validation for more robust performance estimates
- Engineer additional features (e.g. car age from `year`, price-per-mile efficiency)
- Handle outliers in `price` and `mileage` more rigorously
- Deploy the model as a simple price-prediction web app (similar to the Heart Disease Prediction project)

## 👤 Author

**Talha Rashid**

*With help and guidance from **Zahid Hashmi***

---
⭐ If you found this project useful, consider giving it a star!

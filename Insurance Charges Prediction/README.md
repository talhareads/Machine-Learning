# 🏥 Insurance Charges Prediction — Machine Learning & EDA

A complete **Machine Learning and Exploratory Data Analysis (EDA)** project for predicting medical insurance charges using demographic, lifestyle, and health-related features.

This project covers the complete workflow from **data exploration and visualization** to **data preprocessing, feature engineering, feature selection, scaling, Linear Regression, and model evaluation**.

---

## 📌 Project Overview

The objective of this project is to analyze an insurance dataset and build a **Linear Regression model** capable of predicting insurance charges based on factors such as:

* Age
* Gender
* BMI
* Number of children
* Smoking status
* Region
* BMI category

The project also explores relationships between the available variables and insurance charges through statistical analysis and visualizations.

---

## 🎯 Objectives

The main objectives of this project are:

1. Understand the structure of the insurance dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Analyze numerical and categorical variables.
4. Identify potential outliers.
5. Analyze correlations between numerical features.
6. Clean and preprocess the dataset.
7. Convert categorical variables into numerical form.
8. Apply One-Hot Encoding to multi-category variables.
9. Perform feature engineering using BMI categories.
10. Scale selected numerical features.
11. Select relevant features using correlation.
12. Train a Linear Regression model.
13. Evaluate the model using **R² Score** and **Adjusted R² Score**.

---

## 📊 Dataset

The project uses the `insurance.csv` dataset.

The original dataset contains:

* **1,338 rows**
* **7 columns**

### Original Features

| Feature    | Description                   | Type               |
| ---------- | ----------------------------- | ------------------ |
| `age`      | Age of the individual         | Numerical          |
| `sex`      | Gender of the individual      | Categorical        |
| `bmi`      | Body Mass Index               | Numerical          |
| `children` | Number of children/dependents | Numerical          |
| `smoker`   | Smoking status                | Categorical        |
| `region`   | Residential region            | Categorical        |
| `charges`  | Medical insurance charges     | Numerical / Target |

### Target Variable

`charges`

The model is trained to predict the insurance charges of an individual.

---

# 🔍 Exploratory Data Analysis (EDA)

## 1. Dataset Inspection

The project begins by loading the dataset using Pandas and examining its structure.

The following operations were performed:

* `head()`
* `info()`
* `shape`
* `describe()`
* `isnull().sum()`
* `columns`

### Dataset Shape

```text
1338 rows × 7 columns
```

The dataset contains:

* 4 numerical columns
* 3 categorical columns

---

## 2. Statistical Summary

Descriptive statistics were calculated for the numerical variables.

| Feature  |      Mean |  Minimum |   Maximum |
| -------- | --------: | -------: | --------: |
| Age      |     39.21 |       18 |        64 |
| BMI      |     30.66 |    15.96 |     53.13 |
| Children |      1.09 |        0 |         5 |
| Charges  | 13,270.42 | 1,121.87 | 63,770.43 |

---

## 3. Missing Value Analysis

Missing values were checked using:

```python
df.isnull().sum()
```

### Result

There were **no missing values** in any of the seven original columns.

---

## 4. Numerical Feature Visualization

Histograms with KDE were created for:

* `age`
* `bmi`
* `children`
* `charges`

These visualizations were used to understand the distributions of the numerical variables.

---

## 5. Categorical Feature Analysis

Count plots were created for:

* `sex`
* `smoker`
* `region`

### Gender Distribution

After removing the duplicate:

* Male: **675**
* Female: **662**

### Smoking Status

* Non-smoker: **1,063**
* Smoker: **274**

### Region Distribution

* Southeast: **364**
* Southwest: **325**
* Northwest: **324**
* Northeast: **324**

---

# 📦 Outlier Analysis

Boxplots were created for all numerical variables:

* `age`
* `bmi`
* `children`
* `charges`

The purpose was to visually inspect the numerical variables for potential outliers.

> The notebook performs **outlier visualization/inspection** but does not remove outliers.

---

# 🔗 Correlation Analysis

A correlation heatmap was created using the numerical variables.

The correlation analysis was later used during feature selection.

Some of the important correlations with `charges` were:

| Feature                   | Correlation with Charges |
| ------------------------- | -----------------------: |
| `Is_Smoker`               |               **0.7872** |
| `age`                     |               **0.2983** |
| `BMI_Category_Obese`      |               **0.2003** |
| `bmi`                     |               **0.1962** |
| `BMI_Category_Overweight` |              **-0.1206** |
| `BMI_Category_Normal`     |              **-0.1040** |
| `children`                |               **0.0674** |
| `Is_Female`               |              **-0.0580** |

The strongest relationship with insurance charges in this project was **smoking status**, with a correlation of approximately **0.787**.

---

# 🧹 Data Cleaning & Preprocessing

A copy of the original dataset was created before preprocessing:

```python
df_clean = df.copy()
```

## Duplicate Removal

Duplicate records were removed using:

```python
df_clean.drop_duplicates(inplace=True)
```

The dataset changed from:

```text
1338 rows → 1337 rows
```

Therefore, **1 duplicate record** was removed.

---

# 🔢 Label Encoding

The categorical `sex` variable was converted into binary values:

```text
male   → 0
female → 1
```

The column was then renamed:

```text
sex → Is_Female
```

Similarly, `smoker` was converted into:

```text
no  → 0
yes → 1
```

and renamed:

```text
smoker → Is_Smoker
```

This transformed the categorical variables into numerical features suitable for Machine Learning.

---

# 🌎 One-Hot Encoding

The `region` feature contains four categories:

* Northeast
* Northwest
* Southeast
* Southwest

Since it contains more than two categories, **One-Hot Encoding** was applied.

This generated:

```text
region_northeast
region_northwest
region_southeast
region_southwest
```

---

# 🧠 Feature Engineering

A new feature called **BMI Category** was created from the `bmi` variable.

The BMI ranges used in the project were:

| BMI Range   | Category    |
| ----------- | ----------- |
| 0 – 18.5    | Underweight |
| 18.5 – 24.9 | Normal      |
| 24.9 – 29.9 | Overweight  |
| 29.9+       | Obese       |

The categorical BMI feature was then converted using One-Hot Encoding.

New features included:

```text
BMI_Category_Underweight
BMI_Category_Normal
BMI_Category_Overweight
BMI_Category_Obese
```

---

# ⚖️ Feature Scaling

`StandardScaler` from Scikit-learn was applied to:

```text
age
bmi
children
```

The purpose of scaling was to transform these numerical features to a standardized scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_clean[cols] = scaler.fit_transform(df_clean[cols])
```

---

# 🎯 Feature Selection

Feature selection was performed using the **correlation of each feature with the target variable (`charges`)**.

A threshold of:

```text
|Correlation| < 0.05
```

was used to identify features for removal.

According to the notebook:

### Dropped Features

```text
region_northwest
region_southwest
```

because their absolute correlation with `charges` was below `0.05`.

### Final Features

The final dataset used for model training contained:

```text
age
Is_Female
bmi
children
Is_Smoker
region_southeast
BMI_Category_Normal
BMI_Category_Overweight
BMI_Category_Obese
charges
```

---

# 🤖 Machine Learning Model

## Linear Regression

The project uses **Linear Regression** for predicting insurance charges.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
```

### Target

```text
charges
```

### Input Features

The model uses the selected features after preprocessing and feature selection.

---

# ✂️ Train-Test Split

The dataset was divided into training and testing sets using:

```python
train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42
)
```

### Split

* **80% Training Data**
* **20% Testing Data**
* `random_state = 42`

This allows the model to learn from the training data and evaluate its performance on unseen test data.

---

# 📈 Model Evaluation

## R² Score

The model achieved:

```text
R² Score = 0.8034
```

An R² score of approximately **0.80** indicates that the Linear Regression model explains a substantial portion of the variation in insurance charges on the test data.

---

## Adjusted R² Score

The project also calculates the Adjusted R² score.

```text
Adjusted R² = 0.7965
```

### Final Results

| Metric      |      Score |
| ----------- | ---------: |
| R² Score    | **0.8034** |
| Adjusted R² | **0.7965** |

---

# 🛠️ Technologies & Libraries

The project was developed using Python and the following libraries:

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 📊 Matplotlib
* 📈 Seaborn
* 🤖 Scikit-learn

### Machine Learning Components

* `train_test_split`
* `LinearRegression`
* `StandardScaler`
* `r2_score`

---

# 📂 Project Structure

```text
Insurance-Prediction/
│
├── Insurance.ipynb
├── insurance.csv
└── README.md
```

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Insurance-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Insurance-Prediction
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
Insurance.ipynb
```

Make sure `insurance.csv` is located in the same directory as the notebook.

---

# 💡 Key Findings

Based on the analysis performed in this project:

* The dataset contains **1,338 original records** and **7 features**.
* No missing values were found.
* One duplicate record was removed during preprocessing.
* Smoking status had the strongest correlation with insurance charges among the analyzed features.
* Age and BMI also showed positive relationships with insurance charges.
* BMI was transformed into meaningful categories through feature engineering.
* Categorical variables were converted into numerical representations.
* Standard scaling was applied to selected numerical features.
* Correlation-based feature selection was used to reduce less relevant features.
* Linear Regression achieved an **R² score of 0.8034**.
* The Adjusted R² score was **0.7965**.

---

# 📌 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Exploratory Data Analysis
   ↓
Missing Value Check
   ↓
Distribution Analysis
   ↓
Categorical Analysis
   ↓
Outlier Visualization
   ↓
Correlation Analysis
   ↓
Duplicate Removal
   ↓
Label Encoding
   ↓
One-Hot Encoding
   ↓
Feature Engineering
   ↓
Feature Scaling
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Linear Regression
   ↓
Prediction
   ↓
R² & Adjusted R² Evaluation
```

---

# 🔮 Future Improvements

Possible improvements for future versions of this project include:

* Testing additional regression algorithms.
* Comparing Linear Regression with Random Forest, Gradient Boosting, and other models.
* Using MAE and RMSE along with R².
* Performing cross-validation.
* Optimizing model hyperparameters where applicable.
* Creating a prediction interface using Streamlit.
* Deploying the trained model as a web application.

---

# 👨‍💻 Author

**Talha Rashid**

BS Artificial Intelligence Student
Institute of Management Sciences (IMSciences), Peshawar

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* Data Science
* Python
* Exploratory Data Analysis
* Predictive Modeling

---

⭐ If you find this project useful, consider giving the repository a star!


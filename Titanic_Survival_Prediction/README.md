# 🚢 Titanic Survival Prediction

A classic machine learning classification project that predicts passenger survival on the Titanic, using the built-in Seaborn Titanic dataset. Multiple algorithms were trained and compared to find the most accurate model.

## 📌 Project Overview

This project explores the well-known Titanic dataset to predict whether a passenger survived, based on features like class, sex, age, fare, and family size. Rather than settling for a single model, **five different classification algorithms** were trained and evaluated to identify the best performer, with cross-validation used to confirm the results.

**Workflow:**
1. Load the built-in Titanic dataset (via `seaborn.load_dataset`)
2. Data cleaning (drop redundant columns, handle missing values)
3. Encode categorical features
4. Train and evaluate 5 classification algorithms
5. Validate the best model using 5-fold cross-validation

## 📊 Dataset

The dataset was loaded directly from **Seaborn's built-in datasets** (`sns.load_dataset('titanic')`) — no external file needed.

**Data cleaning steps:**
- Dropped redundant/duplicate-information columns: `deck`, `embark_town`, `alive`, `class`, `who`, `adult_male`
- Filled missing `age` values with the column mean
- Dropped rows with missing `embarked` values

**Features used for prediction:**

| Column | Description |
|---|---|
| `pclass` | Passenger class (1st, 2nd, 3rd) |
| `sex` | Gender (label encoded) |
| `age` | Age (missing values filled with mean) |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Ticket fare |
| `embarked` | Port of embarkation (label encoded) |
| `alone` | Whether the passenger was traveling alone (converted to int) |

**Target:** `survived` (0 = did not survive, 1 = survived)

## 🤖 Models Trained & Results

Five classification algorithms were trained on an 80/67 train-test split (`test_size=0.33`, `random_state=42`), with feature scaling (StandardScaler) applied for the distance/margin-based models (KNN, Decision Tree, SVM):

| Model | Accuracy |
|---|---|
| **Logistic Regression** | **0.8163** |
| **SVM (RBF kernel)** | **0.8163** |
| K-Nearest Neighbors | 0.8027 |
| Naive Bayes | 0.7789 |
| Decision Tree | 0.7585 |

**Logistic Regression and SVM tied as the top performers at 81.6% accuracy.**

### ✅ Final Model Validation

To confirm which model generalizes better, **5-fold cross-validation** was run on SVM:

```
Fold scores: [0.8315, 0.8202, 0.8146, 0.8090, 0.8644]
Mean CV accuracy: 0.8279 (≈ 82.8%)
```

This confirmed **SVM (RBF kernel)** as the most robust and accurate model for this dataset, with consistent performance across folds.

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — data loading & visualization
- **Scikit-learn** — preprocessing, model training (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM), evaluation, cross-validation

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── Titanic_Survival_Prediction.ipynb   # EDA, preprocessing, model training & comparison
└── README.md
```

## 🚀 Future Improvements

- Perform hyperparameter tuning (`GridSearchCV`) on SVM to push accuracy further
- Engineer new features (e.g. title extracted from name, family size from `sibsp` + `parch`)
- Try ensemble methods (Random Forest, XGBoost, Voting Classifier)
- Deploy the final model as an interactive web app (similar to the Heart Disease Prediction project)

## 👤 Author

**Talha Rashid**

*With help and guidance from **Zahid Hashmi***

---
⭐ If you found this project useful, consider giving it a star!

# 🌸 Ensemble Learning — Iris Classification

A machine learning project exploring and comparing the three major ensemble learning techniques — **Stacking, Bagging, and Boosting** — using the built-in Iris dataset.

## 📌 Project Overview

Instead of relying on a single classifier, this project demonstrates how combining multiple models through ensemble techniques can improve (or match) prediction performance. Five different ensemble approaches were implemented and evaluated on the classic Iris flower classification problem.

**Workflow:**
1. Load the built-in Iris dataset (via `seaborn.load_dataset`)
2. Encode the target labels
3. Implement and evaluate **Stacking** (multiple base learners + meta-model)
4. Implement and evaluate **Bagging** (Random Forest)
5. Implement and evaluate **Boosting** (AdaBoost, Gradient Boosting, XGBoost)
6. Compare accuracy across all techniques

## 📊 Dataset

The dataset was loaded directly from **Seaborn's built-in datasets** (`sns.load_dataset('iris')`) — no external file needed.

- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Target:** `species` (label encoded — Setosa, Versicolor, Virginica)
- **Split:** 80/20 train-test split (`test_size=0.2`, `random_state=42`)

## 🤖 Ensemble Techniques Implemented

### 1️⃣ Stacking
Combines multiple base learners and uses a meta-model to make the final prediction.
- **Base models:** Decision Tree, SVM (RBF kernel, probability=True), Logistic Regression
- **Meta-model:** Logistic Regression
- **Cross-validation:** 5-fold (used internally by `StackingClassifier`)

### 2️⃣ Bagging
Reduces variance by training multiple models on random subsets of data.
- **Model:** Random Forest (`n_estimators=100`, `max_features='sqrt'`)

### 3️⃣ Boosting
Builds models sequentially, with each new model correcting the errors of the previous one.
- **AdaBoost** (`n_estimators=100`)
- **Gradient Boosting** (`n_estimators=100`, `learning_rate=0.1`)
- **XGBoost** (`learning_rate=0.1`, `max_depth=3`)

## 📈 Results

| Technique | Model | Accuracy |
|---|---|---|
| Stacking | DT + SVM + LR → Logistic Regression | **1.0000** |
| Bagging | Random Forest | **1.0000** |
| Boosting | AdaBoost | 0.9333 |
| Boosting | Gradient Boosting | **1.0000** |
| Boosting | XGBoost | **1.0000** |

**Stacking, Random Forest (Bagging), Gradient Boosting, and XGBoost all achieved perfect accuracy (100%)** on the test set, while AdaBoost slightly trailed at 93.3%. Since Iris is a small, well-separated, and relatively easy dataset, this high accuracy is expected — the real value of this project lies in implementing and comparing all three ensemble paradigms end-to-end.

## 🛠️ Tech Stack

- **Python**
- **Pandas** — data handling
- **Seaborn** — dataset loading
- **Scikit-learn** — Stacking, Random Forest, AdaBoost, Gradient Boosting, preprocessing & evaluation
- **XGBoost** — gradient-boosted trees

## 📁 Project Structure

```
ensemble-learning-iris-classification/
│
├── Ensemble_Learning_Iris_Classification.ipynb   # Stacking, Bagging & Boosting implementation
└── README.md
```

## 🚀 Future Improvements

- Test these ensemble techniques on a larger, noisier, real-world dataset where differences between models would be more visible
- Add cross-validation scoring for all models (not just Stacking) for a fairer comparison
- Visualize decision boundaries for each ensemble technique
- Tune hyperparameters (`GridSearchCV`) for AdaBoost to close the accuracy gap

## 👤 Author

**Talha Rashid**

*With help and guidance from **Zahid Hashmi***

---
⭐ If you found this project useful, consider giving it a star!

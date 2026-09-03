# 🎯 Hyperparameter Tuning — GridSearchCV & RandomizedSearchCV

A machine learning project focused on **hyperparameter tuning** techniques, comparing baseline models against `GridSearchCV` and `RandomizedSearchCV` optimization using the built-in Iris dataset.

## 📌 Project Overview

Choosing the right model is only half the job — tuning its hyperparameters is what separates a good model from an optimal one. This project demonstrates how to systematically search for the best hyperparameters using two of Scikit-learn's most widely used tuning strategies: **Grid Search** (exhaustive) and **Randomized Search** (sampled).

**Workflow:**
1. Load the built-in Iris dataset (via `seaborn.load_dataset`)
2. Train baseline **KNN** and **SVM** models
3. Tune **KNN** using `GridSearchCV`
4. Tune **SVM** using `GridSearchCV`
5. Tune **SVM** using `RandomizedSearchCV` for comparison

## 📊 Dataset

The dataset was loaded directly from **Seaborn's built-in datasets** (`sns.load_dataset('iris')`) — no external file needed.

- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Target:** `species` (3 classes — Setosa, Versicolor, Virginica)
- **Split:** 80/20 train-test split (`test_size=0.2`, `random_state=42`) for baseline models; full dataset used with 5-fold CV for tuning

## 🤖 Baseline Models

| Model | Default Test Accuracy |
|---|---|
| K-Nearest Neighbors (`n_neighbors=5`) | 1.0000 |
| SVM (default RBF) | 1.0000 |

Both baseline models already performed perfectly on the hold-out test split — the tuning process was then used to find the most **robust** configuration via cross-validation, rather than just chasing a higher single-split score.

## 🔍 Hyperparameter Tuning Results

### 1️⃣ GridSearchCV — K-Nearest Neighbors
Searched over:
- `n_neighbors`: [3, 5, 7, 9]
- `weights`: ['uniform', 'distance']
- `algorithm`: ['auto', 'ball_tree', 'kd_tree', 'brute']

**Best cross-validated accuracy: 0.9800** (achieved with `n_neighbors=7, weights='uniform'`, consistent across `algorithm` choices — rank 1 in `cv_results_`).

### 2️⃣ GridSearchCV — Support Vector Machine
Searched over:
- `kernel`: ['linear', 'rbf']
- `C`: [0.1, 1, 10]
- `gamma`: ['scale', 'auto']

**Best cross-validated accuracy: 0.9800**, achieved by multiple configurations including `C=1, kernel='linear'` and `C=10, kernel='rbf'`.

### 3️⃣ RandomizedSearchCV — Support Vector Machine
Same parameter space as above, sampled with `n_iter=4` (instead of exhaustively testing all 12 combinations).

| C | Kernel | Mean Test Score |
|---|---|---|
| **10** | **rbf** | **0.9800** |
| 10 | rbf | 0.9800 |
| 0.1 | linear | 0.9733 |
| 0.1 | rbf | 0.9467 |

RandomizedSearchCV found the **same best score (0.98)** as the exhaustive GridSearchCV, but using only 4 sampled combinations instead of all 12 — demonstrating how randomized search can find near-optimal hyperparameters much faster, which becomes especially valuable on larger search spaces or datasets.

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** — data handling
- **Seaborn** — dataset loading
- **Scikit-learn** — KNN, SVM, `GridSearchCV`, `RandomizedSearchCV`, cross-validation

## 📁 Project Structure

```
hyperparameter-tuning-gridsearch-randomsearch/
│
├── Hyperparameter_Tuning_GridSearch_RandomSearch.ipynb   # Baseline models + GridSearch + RandomSearch
└── README.md
```

## 🚀 Future Improvements

- Apply the same tuning workflow to a larger, more complex dataset where the value of tuning is more visible
- Extend tuning to additional models (Random Forest, XGBoost)
- Visualize the hyperparameter search space (heatmaps of `C` vs `gamma` vs accuracy)
- Compare tuning runtime between GridSearchCV and RandomizedSearchCV explicitly

## 👤 Author

**Talha Rashid**

*With help and guidance from **Zahid Hashmi***

---
⭐ If you found this project useful, consider giving it a star!

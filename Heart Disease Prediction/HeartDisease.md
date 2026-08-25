# ❤️ Heart Disease Prediction

An end-to-end machine learning project that predicts a patient's risk of heart disease based on clinical and demographic features, with an interactive web app built using **Streamlit**.

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide, and early detection can significantly improve patient outcomes. This project uses a real-world clinical dataset to explore the relationships between patient health indicators and heart disease, then trains a classification model to predict risk. The final model is deployed as a simple, interactive web app where users can input their health parameters and instantly get a prediction.

**Workflow:**
1. Exploratory Data Analysis (EDA) on patient health data
2. Data preprocessing (encoding categorical features, scaling numerical features)
3. Model training (Logistic Regression)
4. Model deployment via a Streamlit web app

## 📊 Dataset

The dataset (`HeartDisease.csv`) contains **918 patient records** with the following features:

| Column | Description | Type |
|---|---|---|
| `Age` | Age of the patient | Numeric |
| `Sex` | Sex of the patient (M / F) | Categorical |
| `ChestPainType` | Chest pain type (ATA, NAP, ASY, TA) | Categorical |
| `RestingBP` | Resting blood pressure (mm Hg) | Numeric |
| `Cholesterol` | Serum cholesterol (mg/dl) | Numeric |
| `FastingBS` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | Binary |
| `RestingECG` | Resting electrocardiogram results (Normal, ST, LVH) | Categorical |
| `MaxHR` | Maximum heart rate achieved | Numeric |
| `ExerciseAngina` | Exercise-induced angina (Y / N) | Categorical |
| `Oldpeak` | ST depression induced by exercise | Numeric |
| `ST_Slope` | Slope of the peak exercise ST segment (Up, Flat, Down) | Categorical |
| `HeartDisease` | Target variable (1 = heart disease, 0 = normal) | Binary |

## 🔍 Exploratory Data Analysis

The EDA phase (`Untitled-1.ipynb`) covers:
- Structural checks: shape, dtypes, missing values, and duplicate rows
- Distribution plots (histograms with KDE) for `Age`, `RestingBP`, `Cholesterol`, and `MaxHR`
- Target balance check via a bar plot of `HeartDisease` counts
- Categorical breakdowns of heart disease occurrence by `Sex`, `ChestPainType`, and `FastingBS`
- A boxplot of `Cholesterol` vs `HeartDisease` and a violin plot of `Age` vs `HeartDisease`
- A correlation heatmap across all numeric features

**Data cleaning:** `RestingBP` and `Cholesterol` contained invalid `0` values (physiologically impossible), which were replaced with the column mean and rounded to 2 decimal places.

**Key findings:**
- Patients with **asymptomatic (ASY) chest pain** and a **flat or down-sloping ST segment** showed a noticeably higher rate of heart disease.
- The target variable is reasonably balanced overall, with a slightly higher proportion of positive (heart disease) cases in this dataset.
- `Oldpeak`, `ST_Slope`, `ChestPainType`, and `MaxHR` showed the strongest correlations with the target, based on the correlation heatmap and a feature-correlation table computed during feature selection.

## 🤖 Model

**Preprocessing & feature engineering:**
- Categorical features one-hot encoded via `pd.get_dummies`
- Numeric features (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) standardized with `StandardScaler`
- Feature selection: computed the correlation of every feature with `HeartDisease` and dropped any feature with |correlation| < 0.05

**Models compared** (80/67 train/test split, `test_size=0.33`, `random_state=42`):

| Model | Accuracy | F1 Score |
|---|---|---|
| **Logistic Regression** | **0.8614** | **0.8772** |
| Naive Bayes | 0.8515 | 0.8665 |
| SVM | 0.8482 | 0.8678 |
| K-Nearest Neighbors | 0.8350 | 0.8503 |
| Decision Tree | 0.7360 | 0.7590 |

**Logistic Regression** was selected as the final model, achieving the best accuracy (**86.1%**) and F1 score (**0.877**) among the models tested.

**Artifacts saved for deployment:**
- `LogReg_Heart.pkl` — trained Logistic Regression model
- `scaler_Heart.pkl` — fitted StandardScaler
- `col_Heart.pkl` — column order used during training (to align inference input)

## 🖥️ Web App

The app (`App.py`) is a **Streamlit** interface that lets a user enter their health details and get an instant prediction.

**Inputs collected:**
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate

**Output:** A clear risk message — ⚠️ *"High Risk of Heart Stroke"* or ✅ *"Low Risk of Heart Stroke"*.

### Run locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run App.py
```

The app will open in your browser at `http://localhost:8501`.

## 📁 Project Structure

```
heart-disease-prediction/
│
├── HeartDisease.csv       # Dataset
├── Untitled-1.ipynb       # EDA, data cleaning & model training notebook
├── App.py                 # Streamlit web app
├── LogReg_Heart.pkl       # Trained model
├── scaler_Heart.pkl       # Fitted scaler
├── col_Heart.pkl          # Training column order
├── requirements.txt       # Project dependencies
└── README.md
```

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — EDA visualizations
- **Scikit-learn** — model training & preprocessing
- **Streamlit** — web app deployment
- **Joblib** — model serialization

## 🚀 Future Improvements

- Try ensemble models (Random Forest, XGBoost) which often outperform Logistic Regression on this type of tabular data
- Add hyperparameter tuning (e.g. `GridSearchCV`) to squeeze out more performance
- Add SHAP/feature importance explanations to the app
- Deploy the app publicly (e.g. Streamlit Community Cloud)

## 👤 Author

**Talha Rashid**

---
⭐ If you found this project useful, consider giving it a star!


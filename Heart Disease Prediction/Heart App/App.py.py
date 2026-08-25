import streamlit as st
import pandas as pd
import joblib
import os

# Get the folder where this Python file is located
path = os.path.dirname(os.path.abspath(__file__))

# Load the saved model, scaler and columns
model = joblib.load(os.path.join(path, 'LogReg_Heart.pkl'))
scaler = joblib.load(os.path.join(path, 'scaler_Heart.pkl'))
column = joblib.load(os.path.join(path, 'col_Heart.pkl'))

st.title("Heart Stroke Prediction")

st.markdown("This app predicts the likelihood of a heart stroke based on user input.")

age = st.slider("Age", 10, 100, 20)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

ChestPain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "TA", "NAP", "ASY"]
)

RestingBP = st.number_input(
    "Resting Blood Pressure",
    80,
    200,
    120
)

Cholesterol = st.number_input(
    "Cholesterol",
    100,
    600,
    200
)

FastingBS = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["Yes", "No"]
)

RestingECG = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

MaxHR = st.number_input(
    "Max Heart Rate",
    60,
    220,
    150
)


if st.button("Predict"):

    input_Data = {
        'Age': age,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'MaxHR': MaxHR,
        'Sex' + sex: 1,
        'ChestPain' + ChestPain: 1,
        'FastingBS': 1 if FastingBS == "Yes" else 0,
        'RestingECG' + RestingECG: 1
    }

    input_df = pd.DataFrame([input_Data])

    # Add missing columns
    for col in column:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in the same order as training data
    input_df = input_df[column]

    # Scale the input data
    input_df = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("High Risk of Heart Stroke.")
    else:
        st.success("Low Risk of Heart Stroke.")


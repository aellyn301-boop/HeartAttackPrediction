import streamlit as st
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

def main():
    st.title('Heart Attack Prediction')

    age = st.number_input("Age", 20, 100)
    sex = st.selectbox("Sex", [0,1])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate")
    exang = st.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = st.number_input("Oldpeak")
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Number of Major Vessels", [0,1,2,3,4])
    thal = st.selectbox("Thal", [0,1,2,3])
    
    if st.button("Predict"):

        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,slope, ca, thal]

        prediction = make_prediction(input_data)

        if prediction == 1:
            st.error("Patient has heart disease")
        else:
            st.success("Patient does NOT have heart disease")

def make_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    continuous_idx = [0, 3, 4, 7, 9]
    numeric_data = input_array[:, continuous_idx]
    scaled_numeric = scaler.transform(numeric_data)
    input_array[:, continuous_idx] = scaled_numeric
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
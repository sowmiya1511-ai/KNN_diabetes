import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

model = joblib.load("diabetes_knn_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")
st.title("🩺 Diabetes Prediction App ")
st.write("This app predicts whether a person has diabetes based on medical measurements.")


menu = st.sidebar.selectbox("Choose an option", 
                            ["Predict Diabetes", "Dataset Preview", "Correlation Heatmap"])


if menu == "Predict Diabetes":
    st.header("🔍 Enter Patient Details")

    Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    Glucose = st.number_input("Glucose Level", min_value=0.0, step=0.1)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1)
    Insulin = st.number_input("Insulin Level", min_value=0.0, step=0.1)
    BMI = st.number_input("BMI", min_value=0.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
    Age = st.number_input("Age", min_value=0, step=1)

    if st.button("Predict"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])

        
        input_scaled = scaler.transform(input_data)

       
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f" High risk of Diabetes! Probability: {probability:.2f}")
        else:
            st.success(f" Low risk of Diabetes. Probability: {probability:.2f}")


elif menu == "Dataset Preview":
    st.header("📄 Dataset Preview")

    df = pd.read_csv("diabetes.csv")
    st.dataframe(df)

    st.write("Shape:", df.shape)


elif menu == "Correlation Heatmap":
    st.header("📊 Correlation Heatmap")

    df = pd.read_csv("diabetes.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

import streamlit as st
import joblib

model = joblib.load("model.pkl")

# Set page title and basic header
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.title("Diabetes Prediction Dashboard")
st.write("Welcome! Enter your details below to predict diabetes risk.")

# Example input widgets
age = st.number_input("Age")
hba1c = st.number_input("HbA1c")
diastolic_bp = st.number_input("Diastolic BP")
systolic_bp = st.number_input("systolic BP")

# Example action button
if st.button("Predict"):
    prediction = model.predict([[systolic_bp,diastolic_bp,hba1c,age]])
    if prediction[0] == 0:
        st.success("Prediction:No Diabetes")
    else:
        st.error("Prediction : Diabetes")
        
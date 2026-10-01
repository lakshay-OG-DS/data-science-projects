
import streamlit as st
import joblib
import numpy as np

model = joblib.load('ev_model.pkl')

st.title("EV Purchase Prediction Portal")
st.write("Enter the customer details below to evaluate EV adoption probability.")

annual_income = st.number_input("Annual Income (USD)", min_value=0.0, value=75000.0, step=1000.0)
stations_home = st.number_input("Charging Stations Near Home", min_value=0, max_value=50, value=3)
stations_work = st.number_input("Charging Stations Near Work", min_value=0, max_value=50, value=2)
env_concern = st.slider("Environmental Concern Level (1 to 5)", min_value=1, max_value=5, value=3)

if st.button("Predict Purchase Intent"):
    input_data = np.array([[annual_income, stations_home, stations_work, env_concern]])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    classes = model.classes_
    class_idx = list(classes).index(prediction)
    confidence = float(probabilities[class_idx])
    
    st.subheader("Results")
    st.write(f"Prediction: {prediction}")
    st.write(f"Model Confidence: {confidence * 100:.2f}%")
    st.progress(int(confidence * 100))
    

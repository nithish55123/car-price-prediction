import streamlit as st
import joblib
import numpy as np
import os

MODEL_PATH = "models/rf_model.joblib"

st.title("Car Price Predictor — Demo")

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Run src/train_model.py first to generate the model.")
else:
    model = joblib.load(MODEL_PATH)

    age = st.number_input("Age (years)", min_value=0, max_value=50, value=5)
    mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
    power = st.number_input("Power (HP)", min_value=10, value=100)

    if st.button("Predict"):
        X = np.array([[age, mileage, power]])
        prediction = model.predict(X)[0]
        st.success(f"Estimated price: ₹{prediction:,.0f}")

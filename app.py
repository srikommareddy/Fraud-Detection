#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing pipeline
model = joblib.load("xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Set Streamlit app title
st.title("💳 Bank Transaction Fraud Detector")

st.markdown("Enter transaction details below to predict whether it is **fraudulent or normal**.")

# --- Input fields ---
TransactionAmount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
TransactionType = st.selectbox("Transaction Type", ["Purchase", "Withdrawal", "Transfer", "Payment"])  # adjust as per data
Location = st.selectbox("Transaction Location", ["City", "Suburban", "Rural"])  # adjust as per data
Channel = st.selectbox("Transaction Channel", ["Online", "ATM", "Branch"])  # adjust as per data
CustomerAge = st.slider("Customer Age", 18, 100, step=1)
CustomerOccupation = st.selectbox("Customer Occupation", ["Engineer", "Teacher", "Doctor", "Student", "Business"])  # adjust
TransactionDuration = st.number_input("Transaction Duration (in seconds)", min_value=0.0, format="%.2f")
LoginAttempts = st.slider("Login Attempts", min_value=0, max_value=10, step=1)
AccountBalance = st.number_input("Account Balance", min_value=0.0, format="%.2f")

# --- Predict button ---
if st.button("Predict Fraud Status"):
    # Collect input into DataFrame
    input_data = pd.DataFrame([{
        "TransactionAmount": TransactionAmount,
        "TransactionType": TransactionType,
        "Location": Location,
        "Channel": Channel,
        "CustomerAge": CustomerAge,
        "CustomerOccupation": CustomerOccupation,
        "TransactionDuration": TransactionDuration,
        "LoginAttempts": LoginAttempts,
        "AccountBalance": AccountBalance
    }])

    # Preprocess the input data
    input_transformed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_transformed)[0]
    prediction_label = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Normal Transaction"

    # Display result
    st.subheader(f"Prediction: {prediction_label}")


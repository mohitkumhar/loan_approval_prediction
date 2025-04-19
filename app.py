import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained pipeline
pipe = joblib.load('model_pipeline.pkl')

st.title("🏦 Loan Approval Prediction App")

st.markdown("### Enter Applicant Details:")

# Raw inputs from user
ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, value=2500.0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=100.0)
Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0.0, value=360.0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])

Gender = st.selectbox("Gender", ['Male', 'Female'])
Married = st.selectbox("Married", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
Property_Area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
Area_History = st.selectbox("Area History", ['Urban', 'Semiurban', 'Rural'])

# Derived features
DTI = CoapplicantIncome / (ApplicantIncome + 1)
EMI_ratio = (LoanAmount / (Loan_Amount_Term + 1)) / (ApplicantIncome + 1)
Loan_per_Applicant = LoanAmount / (1 + (CoapplicantIncome > 0))
Log_ApplicantIncome = np.log1p(ApplicantIncome)
Log_LoanAmount = np.log1p(LoanAmount)
# Income_bin cannot be reliably computed from single input; default to 0
Income_bin = 0

# Prepare DataFrame for prediction
data = {
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'DTI': DTI,
    'EMI_ratio': EMI_ratio,
    'Loan_per_Applicant': Loan_per_Applicant,
    'Income_bin': Income_bin,
    'Log_ApplicantIncome': Log_ApplicantIncome,
    'Log_LoanAmount': Log_LoanAmount,
    'Gender': Gender,
    'Married': Married,
    'Dependents': Dependents,
    'Education': Education,
    'Self_Employed': Self_Employed,
    'Property_Area': Property_Area,
    'Area_History': Area_History
}
input_df = pd.DataFrame([data])

if st.button("Predict"):
    prediction = pipe.predict(input_df)[0]
    prob = pipe.predict_proba(input_df)[0][1]

    st.markdown("## 🔍 Prediction Result:")
    if prediction == 1:
        st.success(f"Loan is likely to be approved! (Confidence: {prob:.2%})")
    else:
        st.error(f"Loan is likely to be rejected. (Confidence: {1 - prob:.2%})")

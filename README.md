# Loan Approval Prediction 🎯

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://loanapprovalprediction-project.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/mohitkumhar/loan_approval_prediction)

This is a Machine Learning-based web application built using **Streamlit** that predicts whether a loan application will be approved or not, based on applicant and loan details. It uses a trained **Random Forest Classifier** and automated preprocessing pipeline under the hood.

---

## 🔗 Live App
👉 [Click here to open the app](https://loanapprovalprediction-project.streamlit.app/)

---

## 💻 Features

- Simple, clean, and user-friendly interface
- Automated data preprocessing (scaling + encoding)
- Real-time prediction using a trained ML model
- All inputs and prediction handled on the main screen (no sidebar)

---

## 📊 Input Parameters

The app calculates derived features (like DTI, EMI, etc.) automatically. You only need to enter:

- Gender
- Marital Status
- Dependents
- Education
- Employment Status
- Property Area
- Loan Amount & Term
- Applicant & Coapplicant Incomes
- Credit History
- Area History

---

## 🚀 How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/mohitkumhar/loan_approval_prediction.git
cd loan_approval_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

---

## 🛠 Tech Stack

- Python
- Scikit-learn (v1.1.3)
- Pandas / NumPy
- Streamlit
- Preprocessing via `ColumnTransformer`

---

## 📂 Project Structure

```
├── app.py                 # Streamlit frontend
├── model_pipeline.pkl     # Preprocessing pipeline
├── requirements.txt       # Dependencies
└── README.md              # Project info
```

---

## 📬 Contact

For any queries or suggestions, feel free to reach out via [GitHub Issues](https://github.com/mohitkumhar/loan_approval_prediction/issues).

"""
Heart Disease Prediction System - Streamlit Web App
Loads heart_svc_trained_model.pkl (produced by the notebook) and recreates the
same LabelEncoders (fit on the same fixed category lists) used during training.
For learning purposes only - not for real medical decisions.
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    encoders = {}
    encoders["AgeGroup"] = LabelEncoder().fit(["Young", "Middle", "Senior"])
    encoders["Sex"] = LabelEncoder().fit(["Male", "Female"])
    encoders["ChestPain"] = LabelEncoder().fit(
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    encoders["MaxHeartRate"] = LabelEncoder().fit(["Low", "Medium", "High"])
    return encoders

model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    "Built for **AIC354 Machine Learning Fundamentals** (Assignment 2). "
    "Enter the patient details and the trained Support Vector Classifier (SVC) "
    "will predict whether heart disease is likely. Trained on the UCI Heart Disease "
    "(Cleveland) dataset using only 4 input attributes."
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    age_group = st.selectbox("Age Group", ["Young", "Middle", "Senior"], index=1,
                             help="Young: under 45, Middle: 45-59, Senior: 60 and above")
    sex = st.selectbox("Sex", ["Male", "Female"])
with col2:
    chest_pain = st.selectbox("Chest Pain Type",
                              ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    max_hr = st.selectbox("Maximum Heart Rate", ["Low", "Medium", "High"], index=1,
                          help="Low: under 120, Medium: 120-150, High: above 150")

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({"AgeGroup": [age_group], "Sex": [sex],
                               "ChestPain": [chest_pain], "MaxHeartRate": [max_hr]})
    encoded = pd.DataFrame({col: encoders[col].transform(user_input[col]) for col in user_input.columns})
    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ NO HEART DISEASE")
    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("Educational demo only. Not a medical diagnosis tool.")

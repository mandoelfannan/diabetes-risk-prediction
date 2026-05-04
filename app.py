import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Title
st.title("🩺 عندك سكر ولا لا")

# Load data
data = pd.read_csv("data/diabetes.csv")

# Cleaning
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].median())

# Split features
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = LogisticRegression(class_weight='balanced')
model.fit(X_scaled, y)

# Inputs
st.sidebar.header("Enter Patient Data")

preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 200, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 30.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 100, 30)

# Prediction
input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

prob = model.predict_proba(input_scaled)[0][1]

# Output
st.subheader("Prediction Result")

if prob > 0.4:
    st.error(f"🔴 High Risk ({prob:.2f})")
elif prob > 0.2:
    st.warning(f"🟠 Medium Risk ({prob:.2f})")
else:
    st.success(f"🟢 Low Risk ({prob:.2f})")
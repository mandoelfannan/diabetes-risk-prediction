# 🩺 Diabetes Risk Prediction App

An end-to-end Machine Learning project that predicts the risk of diabetes based on patient medical data.  
The project includes data preprocessing, exploratory analysis, model training, evaluation, and an interactive web application using Streamlit.

---

## 📌 Project Objective

The goal of this project is to build a predictive system that estimates whether a patient is at risk of developing diabetes based on clinical health indicators.

This helps in early detection and supports medical decision-making.

---

## 🧠 Problem Type

- Binary Classification Problem  
- Target Variable: `Outcome`  
  - 0 → No Diabetes  
  - 1 → Diabetes  

---

## 📊 Dataset Description

The dataset contains medical diagnostic measurements:

- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  
- Outcome  

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Replaced invalid zero values with median values
- Handled missing data
- Applied feature scaling using `StandardScaler`

### 2. Model Training
Multiple machine learning models were tested:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- Gradient Boosting Classifier  

### 3. Evaluation Metrics
- Accuracy  
- Recall  
- Precision  
- F1 Score  
- Confusion Matrix  

### 4. Threshold Tuning
Different probability thresholds were tested (0.5 → 0.4 → 0.3) to improve recall, which is important in medical prediction tasks.

---

## 🏆 Best Model Performance

- Accuracy: ~0.70 – 0.75  
- Recall: Up to ~0.80 – 0.90 (after threshold tuning)  
- F1 Score: ~0.62 – 0.66  

> The model prioritizes Recall to reduce false negatives (important in medical diagnosis).

---

## 🚀 Web Application (Streamlit)

An interactive web application allows users to input patient data and receive real-time predictions.

### Features:
- User-friendly interface  
- Real-time prediction  
- Risk classification (Low / Medium / High)  
- Probability score output  

---

## 🛠️ Tech Stack

- Python 🐍  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- Streamlit  

---

## 📁 Project Structure

diabetes-risk-prediction/
│
├── data/
│ └── diabetes.csv
│
├── notebook/
│ └── diabetes-risk-prediction.ipynb
│
├── src/
│ └── model.py
│
├── app.py
├── requirements.txt
├── README.md


---

## ▶️ How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt

Run Streamlit app
streamlit run app.py

📌 Key Insights
Glucose level is the most important feature in prediction
Recall is prioritized over accuracy due to medical risk sensitivity
Threshold tuning significantly improves detection of positive cases
📈 Future Improvements
Hyperparameter tuning (GridSearchCV)
Model explainability using SHAP
Deployment on cloud (Streamlit Cloud / Render)
Add dashboard visualizations
👨‍💻 Author

Mohamed Shaban
Data Science & AI Enthusiast
Focused on Machine Learning, Data Analysis, and AI Systems

⭐ If you like this project

Give it a star ⭐ and feel free to contribute or improve it.








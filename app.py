import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and prepare the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("student-mat.csv")
    df['average_score'] = df[['G1', 'G2', 'G3']].mean(axis=1)
    df['Final_Result'] = np.where(df['average_score'] >= 10, 'Pass', 'Fail')
    df.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)
    
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    df['Final_Result'] = LabelEncoder().fit_transform(df['Final_Result'])

    return df, label_encoders

df, label_encoders = load_data()

# Split data
X = df.drop('Final_Result', axis=1)
y = df['Final_Result']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# App title
st.title("🎓 AI-Powered Student Performance Predictor")

# User input form
st.header("Enter Student Details")
with st.form("input_form"):
    age = st.slider("Age", 15, 22, 17)
    studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
    absences = st.slider("Total Absences", 0, 30, 3)
    health = st.slider("Health (1=Poor, 5=Excellent)", 1, 5, 3)
    freetime = st.slider("Free Time After School", 1, 5, 3)
    goout = st.slider("Going Out With Friends", 1, 5, 3)
    Walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 1)
    Dalc = st.slider("Weekday Alcohol Consumption (1-5)", 1, 5, 1)
    
    internet = st.radio("Internet Access at Home", ["yes", "no"])
    higher = st.radio("Wants Higher Education", ["yes", "no"])
    activities = st.radio("Attends Extra Activities", ["yes", "no"])

    submit = st.form_submit_button("Predict")

# On form submission
if submit:
    # Create input vector (only using selected features for simplicity)
    input_data = pd.DataFrame([[
        age, studytime, failures, absences, health, freetime, goout,
        1 if Walc else 0, 1 if Dalc else 0,
        1 if internet == "yes" else 0,
        1 if higher == "yes" else 0,
        1 if activities == "yes" else 0
    ]], columns=[
        'age', 'studytime', 'failures', 'absences', 'health',
        'freetime', 'goout', 'Walc', 'Dalc',
        'internet', 'higher', 'activities'
    ])

    # Add default values for missing required features
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = X[col].mean()

    # Match column order
    input_data = input_data[X.columns]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    result = "🎉 Pass" if prediction == 1 else "❌ Fail"

    st.subheader("Prediction Result")
    st.success(f"The student is predicted to: **{result}**")

import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Title
st.title("🎓 Student Dropout Predictor using KNN")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("student_performance_dataset.csv.xlsx")
    return df

df = load_data()

# Model Training
X = df[['Attendance', 'Internal_Marks']]
y = df['Dropped_Out']
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# User Input
st.subheader("📥 Enter Student Data")
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=75)
internal_marks = st.slider("Internal Marks (%)", min_value=0, max_value=100, value=70)

if st.button("Predict Dropout Risk"):
    input_data = pd.DataFrame([[attendance, internal_marks]], columns=['Attendance', 'Internal_Marks'])
    prediction = knn.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Student is at risk of dropping out.")
    else:
        st.success("✅ Student is likely to continue.")

# Show dataset
if st.checkbox("📊 Show Training Dataset"):
    st.dataframe(df)

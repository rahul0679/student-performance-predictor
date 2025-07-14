import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Load the dataset
df = pd.read_excel("student_performance_dataset.csv.xlsx")

# Features and Labels
X = df[['Attendance', 'Internal_Marks']]
y = df['Dropped_Out']

# Initialize KNN with K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X, y)

# New Student Data
new_student = pd.DataFrame([[70, 60]], columns=['Attendance', 'Internal_Marks'])

# Predict dropout risk
prediction = knn.predict(new_student)
print("Prediction for new student (1 = Dropout, 0 = No Dropout):", prediction[0])
